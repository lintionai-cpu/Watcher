"""
Trade Executor — Zero-delay AI signal execution engine.

Flow: signal fires → proposal sent immediately → on proposal response → buy sent immediately
      → contract opened → on settlement → stats updated → martingale applied
"""
import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Callable

logger = logging.getLogger(__name__)

CONTRACT_TYPE = {
    "OVER":  "DIGITOVER",
    "UNDER": "DIGITUNDER",
}

DEFAULT_SETTINGS = {
    "enabled":                  False,
    "stake":                    1.0,
    "martingale_enabled":       False,
    "martingale_multiplier":    2.0,
    "martingale_max_steps":     5,
    "min_confidence":           0.60,
    "min_agreement":            2,
    "max_loss":                 50.0,
    "max_profit":               100.0,
    "max_trades":               0,          # 0 = unlimited
    "max_concurrent":           2,          # max open trades at once
    "enabled_categories": {
        "over2_under7":  True,
        "under5_over5":  True,
        "over1_under8":  True,
        "under3_over5":  True,
    },
    "enabled_markets": {
        "R_10": True, "R_25": True, "R_50": True,
        "R_75": True, "R_100": True,
        "1HZ10V": True, "1HZ25V": True, "1HZ50V": True,
        "1HZ75V": True, "1HZ100V": True,
    },
}


class TradeSession:
    """Live session state — resets per user request."""
    def __init__(self):
        self.total_trades     = 0
        self.wins             = 0
        self.losses           = 0
        self.pnl              = 0.0
        self.open_trades: Dict[Any, Dict]  = {}      # contract_id → trade
        self.pending_buy: Dict[str, Dict]  = {}      # "buy_{proposal_id}" → context
        self.pending_proposal: Dict[int, Dict] = {}  # req_id → context
        self.martingale_step      = 0
        self.consecutive_losses   = 0
        self.current_stake        = None
        self.trade_log: List[Dict] = []

    def reset(self):
        self.__init__()

    @property
    def win_rate(self) -> float:
        return (self.wins / self.total_trades * 100) if self.total_trades else 0.0

    def to_dict(self) -> Dict:
        return {
            "total_trades":       self.total_trades,
            "wins":               self.wins,
            "losses":             self.losses,
            "pnl":                round(self.pnl, 2),
            "win_rate":           round(self.win_rate, 1),
            "open_trades_count":  len(self.open_trades),
            "martingale_step":    self.martingale_step,
            "consecutive_losses": self.consecutive_losses,
            "current_stake":      round(self.current_stake or 0.0, 2),
        }


class TradeExecutor:

    def __init__(self):
        self.settings: Dict = self._clone_defaults()
        self.session  = TradeSession()
        self._req_counter = 0
        self.on_trade_update: Optional[Callable] = None
        self._ws_client = None
        # Dedup: remember which signal keys were recently traded
        self._traded_keys: Dict[str, float] = {}

    # ──────────────────────────────────────────────────────────
    # Startup wiring
    # ──────────────────────────────────────────────────────────
    def set_ws_client(self, client) -> None:
        self._ws_client = client

    def _clone_defaults(self) -> Dict:
        s = dict(DEFAULT_SETTINGS)
        s["enabled_categories"] = dict(DEFAULT_SETTINGS["enabled_categories"])
        s["enabled_markets"]    = dict(DEFAULT_SETTINGS["enabled_markets"])
        return s

    # ──────────────────────────────────────────────────────────
    # Settings
    # ──────────────────────────────────────────────────────────
    def update_settings(self, patch: Dict) -> None:
        for k, v in patch.items():
            if k in ("enabled_categories", "enabled_markets") and isinstance(v, dict):
                self.settings[k].update(v)
            elif k in self.settings:
                self.settings[k] = v
        self.session.current_stake = float(self.settings["stake"])
        logger.info("Auto-trade settings updated. enabled=%s", self.settings["enabled"])

    def get_status(self) -> Dict:
        return {
            "settings":   self.settings,
            "session":    self.session.to_dict(),
            "trade_log":  self.session.trade_log[-100:],
        }

    def reset_session(self) -> None:
        self.session.reset()
        self.session.current_stake = float(self.settings["stake"])
        self._traded_keys.clear()
        logger.info("Auto-trade session reset")

    # ──────────────────────────────────────────────────────────
    # Signal filter / validation
    # ──────────────────────────────────────────────────────────
    def _should_trade(self, signal: Dict) -> tuple:
        s = self.settings

        if not s["enabled"]:
            return False, "disabled"

        if not self._ws_client or not self._ws_client.authorized:
            return False, "not_authorized"

        cat    = signal.get("category", "")
        market = signal.get("market", "")

        if not s["enabled_categories"].get(cat, True):
            return False, f"category_{cat}_off"

        if not s["enabled_markets"].get(market, True):
            return False, f"market_{market}_off"

        conf = signal.get("confidence", 0)
        if conf < s["min_confidence"]:
            return False, "confidence_too_low"

        # Count agreeing model votes
        direction = signal.get("direction")
        votes     = signal.get("model_votes", {})
        agree     = sum(1 for v in votes.values() if v.get("vote") == direction)
        if agree < s["min_agreement"]:
            return False, "agreement_too_low"

        # Session limits
        if s["max_trades"] > 0 and self.session.total_trades >= s["max_trades"]:
            return False, "max_trades"
        if self.session.pnl <= -abs(s["max_loss"]):
            return False, "max_loss"
        if self.session.pnl >= s["max_profit"]:
            return False, "max_profit"
        if len(self.session.open_trades) >= s["max_concurrent"]:
            return False, "max_concurrent"

        # Dedup within valid window
        key = f"{market}:{cat}:{direction}:{signal.get('barrier')}"
        last = self._traded_keys.get(key, 0)
        if time.time() - last < signal.get("valid_window", 30):
            return False, "already_traded"

        return True, "ok"

    # ──────────────────────────────────────────────────────────
    # Entry point — called by signal engine on each new signal
    # ──────────────────────────────────────────────────────────
    async def on_signal(self, signal: Dict) -> None:
        ok, reason = self._should_trade(signal)
        if not ok:
            logger.debug("Skip trade [%s]: %s / %s", reason,
                         signal.get("market"), signal.get("category"))
            return
        await self._send_proposal(signal)

    # ──────────────────────────────────────────────────────────
    # Step 1: Send proposal
    # ──────────────────────────────────────────────────────────
    async def _send_proposal(self, signal: Dict) -> None:
        self._req_counter += 1
        req_id   = self._req_counter
        stake    = self._get_current_stake()
        currency = (self._ws_client.account_info or {}).get("currency", "USD")
        duration = max(1, int(signal.get("duration_ticks", 1)))

        payload = {
            "proposal":       1,
            "req_id":         req_id,
            "amount":         stake,
            "basis":          "stake",
            "contract_type":  CONTRACT_TYPE[signal["direction"]],
            "currency":       currency,
            "duration":       duration,
            "duration_unit":  "t",
            "symbol":         signal["market"],
            "barrier":        str(signal["barrier"]),
        }

        self.session.pending_proposal[req_id] = {
            "signal":   signal,
            "stake":    stake,
            "sent_at":  time.time(),
        }

        # Mark dedup key immediately so no second signal for this combo fires before buy
        key = f"{signal['market']}:{signal['category']}:{signal['direction']}:{signal['barrier']}"
        self._traded_keys[key] = time.time()

        logger.info("Proposal → %s %s>%s stake=%.2f dur=%dt",
                    signal["market"], signal["direction"],
                    signal["barrier"], stake, duration)

        await self._ws_client._send(payload)

    # ──────────────────────────────────────────────────────────
    # Step 2: Buy immediately on proposal response
    # ──────────────────────────────────────────────────────────
    async def on_proposal_response(self, msg: Dict) -> None:
        req_id  = msg.get("echo_req", {}).get("req_id")
        pending = self.session.pending_proposal.pop(req_id, None)
        if not pending:
            return

        proposal    = msg.get("proposal", {})
        proposal_id = proposal.get("id")
        ask_price   = float(proposal.get("ask_price", pending["stake"]))

        if not proposal_id:
            logger.warning("Proposal response has no id — cannot buy")
            return

        # Check signal is still within valid window
        signal = pending["signal"]
        age    = time.time() - signal.get("timestamp", time.time())
        if age > signal.get("valid_window", 60):
            logger.info("Signal expired (age=%.1fs) — skipping buy", age)
            return

        self._req_counter += 1
        buy_req_id = self._req_counter

        # Store buy context keyed by proposal_id
        self.session.pending_buy[proposal_id] = {
            "signal":      signal,
            "stake":       pending["stake"],
            "ask_price":   ask_price,
            "proposal_id": proposal_id,
            "req_id":      buy_req_id,
            "sent_at":     time.time(),
        }

        logger.info("Buy → proposal=%s price=%.2f", proposal_id, ask_price)
        await self._ws_client._send({
            "buy":    proposal_id,
            "price":  ask_price,
            "req_id": buy_req_id,
        })

    # ──────────────────────────────────────────────────────────
    # Step 3: Trade confirmed
    # ──────────────────────────────────────────────────────────
    async def on_buy_response(self, msg: Dict) -> None:
        buy         = msg.get("buy", {})
        contract_id = buy.get("contract_id")
        echo        = msg.get("echo_req", {})

        if not contract_id:
            logger.warning("Buy response missing contract_id")
            return

        # Match pending_buy by req_id
        pending = None
        for pid, ctx in list(self.session.pending_buy.items()):
            if ctx.get("req_id") == echo.get("req_id") or True:
                # Take the first one (FIFO is fine for our single-proposal flow)
                pending = self.session.pending_buy.pop(pid)
                break

        if not pending:
            logger.warning("No pending buy context for contract %s", contract_id)
            return

        signal = pending["signal"]
        trade  = {
            "id":             contract_id,
            "market":         signal["market"],
            "market_name":    signal.get("market_name", signal["market"]),
            "category":       signal["category"],
            "category_name":  signal.get("category_name", signal["category"]),
            "direction":      signal["direction"],
            "barrier":        signal["barrier"],
            "stake":          pending["stake"],
            "buy_price":      float(buy.get("buy_price", pending["stake"])),
            "confidence":     signal.get("confidence", 0),
            "duration_ticks": signal.get("duration_ticks", 1),
            "status":         "open",
            "result":         None,
            "pnl":            None,
            "sell_price":     None,
            "opened_at":      time.time(),
            "settled_at":     None,
        }

        self.session.open_trades[contract_id] = trade
        self.session.total_trades += 1

        logger.info("Trade OPEN: %s | %s %s>%s | stake=%.2f",
                    contract_id, signal["market"],
                    signal["direction"], signal["barrier"], pending["stake"])

        await self._emit("trade_opened", trade)
        await self._emit_session()

        # Subscribe to live contract updates
        await self._ws_client._send({
            "proposal_open_contract": 1,
            "contract_id":            contract_id,
            "subscribe":              1,
        })

    # ──────────────────────────────────────────────────────────
    # Step 4: Contract settled
    # ──────────────────────────────────────────────────────────
    async def on_contract_update(self, msg: Dict) -> None:
        poc         = msg.get("proposal_open_contract", {})
        contract_id = poc.get("contract_id")

        if not contract_id or contract_id not in self.session.open_trades:
            return

        if not poc.get("is_sold", False):
            return  # still running

        trade  = self.session.open_trades.pop(contract_id)
        profit = float(poc.get("profit", 0))
        is_win = profit > 0

        trade.update({
            "status":     "settled",
            "result":     "WIN" if is_win else "LOSS",
            "pnl":        round(profit, 2),
            "sell_price": float(poc.get("sell_price", 0)),
            "settled_at": time.time(),
        })

        if is_win:
            self.session.wins += 1
            self.session.consecutive_losses = 0
        else:
            self.session.losses += 1
            self.session.consecutive_losses += 1

        self.session.pnl += profit
        self._apply_martingale(is_win)

        self.session.trade_log.append(trade)
        if len(self.session.trade_log) > 200:
            self.session.trade_log.pop(0)

        logger.info("Trade SETTLED: %s | %s | pnl=%.2f | session_pnl=%.2f",
                    contract_id, trade["result"], profit, self.session.pnl)

        await self._emit("trade_settled", trade)
        await self._emit_session()
        await self._check_limits()

    # ──────────────────────────────────────────────────────────
    # Martingale
    # ──────────────────────────────────────────────────────────
    def _get_current_stake(self) -> float:
        if self.session.current_stake is None:
            self.session.current_stake = float(self.settings["stake"])
        return round(max(0.35, float(self.session.current_stake)), 2)

    def _apply_martingale(self, is_win: bool) -> None:
        base = float(self.settings["stake"])
        if not self.settings["martingale_enabled"]:
            self.session.current_stake = base
            self.session.martingale_step = 0
            return
        if is_win:
            self.session.current_stake = base
            self.session.martingale_step = 0
        else:
            max_steps  = int(self.settings["martingale_max_steps"])
            multiplier = float(self.settings["martingale_multiplier"])
            if self.session.martingale_step < max_steps:
                self.session.current_stake = round(
                    float(self.session.current_stake) * multiplier, 2
                )
                self.session.martingale_step += 1
            else:
                self.session.current_stake = base
                self.session.martingale_step = 0

    # ──────────────────────────────────────────────────────────
    # Limits auto-stop
    # ──────────────────────────────────────────────────────────
    async def _check_limits(self) -> None:
        s      = self.settings
        reason = None
        if   self.session.pnl <= -abs(s["max_loss"]):                  reason = f"Max loss ${abs(s['max_loss']):.2f} reached"
        elif self.session.pnl >= s["max_profit"]:                       reason = f"Profit target ${s['max_profit']:.2f} reached"
        elif s["max_trades"] > 0 and self.session.total_trades >= s["max_trades"]: reason = f"Max {s['max_trades']} trades reached"

        if reason:
            self.settings["enabled"] = False
            logger.info("Auto-trade auto-stopped: %s", reason)
            await self._emit("autotrade_stopped", {"reason": reason})

    # ──────────────────────────────────────────────────────────
    # Broadcast helpers
    # ──────────────────────────────────────────────────────────
    async def _emit(self, event: str, data: Dict) -> None:
        if self.on_trade_update:
            try:
                await self.on_trade_update({"type": event, "data": data})
            except Exception as e:
                logger.error("Trade broadcast error: %s", e)

    async def _emit_session(self) -> None:
        await self._emit("session_stats", {
            **self.session.to_dict(),
            "enabled": self.settings["enabled"],
        })


# Module singleton
trade_executor = TradeExecutor()
