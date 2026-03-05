"""
Deriv WebSocket Client Manager — with trade execution message handling.
"""
import asyncio
import json
import time
import logging
from typing import Optional, Callable, Dict, List, Any

import websockets

logger = logging.getLogger(__name__)

DERIV_WSS    = "wss://ws.derivws.com/websockets/v3?app_id=1089"
PING_INTERVAL  = 25
RECONNECT_BASE = 2
RECONNECT_MAX  = 60

MARKETS = {
    "R_10":    "Volatility 10",
    "R_25":    "Volatility 25",
    "R_50":    "Volatility 50",
    "R_75":    "Volatility 75",
    "R_100":   "Volatility 100",
    "1HZ10V":  "Volatility 10 (1s)",
    "1HZ25V":  "Volatility 25 (1s)",
    "1HZ50V":  "Volatility 50 (1s)",
    "1HZ75V":  "Volatility 75 (1s)",
    "1HZ100V": "Volatility 100 (1s)",
}


class DerivWSClient:
    def __init__(self):
        self.ws:              Optional[Any] = None
        self.api_token:       Optional[str] = None
        self.connected:       bool = False
        self.authorized:      bool = False
        self.account_info:    Dict = {}
        self.reconnect_count: int  = 0
        self._running:        bool = False
        self._ping_task:      Optional[asyncio.Task] = None

        # Analysis callbacks
        self.on_tick:           Optional[Callable] = None
        self.on_status_change:  Optional[Callable] = None
        self.on_account_info:   Optional[Callable] = None
        self.on_error:          Optional[Callable] = None

        # Trade execution callbacks (wired from trade_executor)
        self.on_proposal:          Optional[Callable] = None
        self.on_buy:               Optional[Callable] = None
        self.on_contract_update:   Optional[Callable] = None

        # Tick buffers
        self.tick_buffers: Dict[str, List[Dict]] = {m: [] for m in MARKETS}
        self.subscribed_markets: set = set()

    # ─── Connection lifecycle ────────────────────────────────────

    async def connect(self, api_token: str) -> bool:
        self.api_token = api_token
        self._running  = True
        self.reconnect_count = 0
        asyncio.create_task(self._connection_loop())
        return True

    async def disconnect(self) -> None:
        self._running = False
        self.subscribed_markets.clear()
        if self._ping_task:
            self._ping_task.cancel()
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass
        self.ws         = None
        self.connected  = False
        self.authorized = False
        self._notify_status()

    async def _connection_loop(self) -> None:
        while self._running:
            try:
                await self._connect_once()
            except Exception as exc:
                logger.warning("Connection error: %s", exc)
                if not self._running:
                    break
                self.connected  = False
                self.authorized = False
                self._notify_status()
                delay = min(RECONNECT_BASE * (2 ** min(self.reconnect_count, 6)), RECONNECT_MAX)
                self.reconnect_count += 1
                logger.info("Reconnecting in %.1fs (attempt %d)", delay, self.reconnect_count)
                await asyncio.sleep(delay)

    async def _connect_once(self) -> None:
        logger.info("Connecting to Deriv WS...")
        async with websockets.connect(DERIV_WSS, ping_interval=None, close_timeout=5) as ws:
            self.ws        = ws
            self.connected = True
            self._notify_status()
            await self._send({"authorize": self.api_token})
            self._ping_task = asyncio.create_task(self._ping_loop())
            try:
                async for raw in ws:
                    if not self._running:
                        break
                    try:
                        await self._handle_message(json.loads(raw))
                    except Exception as exc:
                        logger.error("Message handling error: %s", exc)
            finally:
                self._ping_task.cancel()
                self.connected  = False
                self.authorized = False
                self._notify_status()

    # ─── Message dispatch ────────────────────────────────────────

    async def _handle_message(self, msg: Dict) -> None:
        mt    = msg.get("msg_type")
        error = msg.get("error")

        if error:
            logger.error("Deriv API error [%s]: %s", mt, error)
            if self.on_error:
                await self._safe(self.on_error, error)
            return

        # ── Account ───────────────────────────────────────────
        if mt == "authorize":
            auth = msg.get("authorize", {})
            self.authorized      = True
            self.reconnect_count = 0
            self.account_info = {
                "loginid":      auth.get("loginid"),
                "balance":      auth.get("balance", 0),
                "currency":     auth.get("currency", "USD"),
                "fullname":     auth.get("fullname", ""),
                "email":        auth.get("email", ""),
                "account_type": auth.get("account_type", ""),
            }
            logger.info("Authorized as %s", self.account_info.get("loginid"))
            self._notify_status()
            if self.on_account_info:
                await self._safe(self.on_account_info, self.account_info)
            for market in MARKETS:
                await self._send({"ticks": market, "subscribe": 1})
                self.subscribed_markets.add(market)
            for market in MARKETS:
                await self._send({
                    "ticks_history": market,
                    "count": 500,
                    "end":   "latest",
                    "style": "ticks",
                })

        # ── Live ticks ────────────────────────────────────────
        elif mt == "tick":
            tick   = msg["tick"]
            market = tick.get("symbol")
            if market in MARKETS:
                price = float(tick["quote"])
                digit = int(str(tick["quote"]).replace(".", "")[-1])
                td = {
                    "market":    market,
                    "price":     price,
                    "digit":     digit,
                    "timestamp": tick.get("epoch", time.time()),
                }
                buf = self.tick_buffers.get(market, [])
                buf.append(td)
                if len(buf) > 2000:
                    buf.pop(0)
                self.tick_buffers[market] = buf
                if self.on_tick:
                    await self._safe(self.on_tick, td)

        # ── Historical ticks ──────────────────────────────────
        elif mt == "history":
            history = msg.get("history", {})
            req     = msg.get("echo_req", {})
            market  = req.get("ticks_history")
            if market and "prices" in history:
                for p, t in zip(history["prices"], history["times"]):
                    price = float(p)
                    digit = int(str(p).replace(".", "")[-1])
                    self.tick_buffers.setdefault(market, []).append({
                        "market": market, "price": price,
                        "digit":  digit,  "timestamp": float(t),
                    })
                self.tick_buffers[market] = sorted(
                    self.tick_buffers[market], key=lambda x: x["timestamp"]
                )[-2000:]
                logger.info("History loaded: %d ticks for %s", len(history["prices"]), market)

        # ── Balance updates ───────────────────────────────────
        elif mt == "balance":
            self.account_info["balance"] = float(msg["balance"]["balance"])
            if self.on_account_info:
                await self._safe(self.on_account_info, self.account_info)

        # ── Trade: proposal response ──────────────────────────
        elif mt == "proposal":
            if self.on_proposal:
                await self._safe(self.on_proposal, msg)

        # ── Trade: buy confirmation ───────────────────────────
        elif mt == "buy":
            if self.on_buy:
                await self._safe(self.on_buy, msg)

        # ── Trade: contract lifecycle ─────────────────────────
        elif mt == "proposal_open_contract":
            if self.on_contract_update:
                await self._safe(self.on_contract_update, msg)

        elif mt == "ping":
            pass

    # ─── Utilities ──────────────────────────────────────────────

    async def _send(self, payload: Dict) -> None:
        if self.ws and self.connected:
            try:
                await self.ws.send(json.dumps(payload))
            except Exception as exc:
                logger.warning("Send failed: %s", exc)

    async def _ping_loop(self) -> None:
        while self._running and self.ws:
            try:
                await asyncio.sleep(PING_INTERVAL)
                if self.ws:
                    await self._send({"ping": 1})
            except (asyncio.CancelledError, Exception):
                break

    def _notify_status(self) -> None:
        if self.on_status_change:
            asyncio.create_task(self._safe(self.on_status_change, {
                "connected":       self.connected,
                "authorized":      self.authorized,
                "reconnect_count": self.reconnect_count,
                "account":         self.account_info,
            }))

    async def _safe(self, fn: Callable, *args) -> None:
        try:
            if asyncio.iscoroutinefunction(fn):
                await fn(*args)
            else:
                fn(*args)
        except Exception as exc:
            logger.error("Callback error: %s", exc)

    # ─── Data access ────────────────────────────────────────────

    def get_digits(self, market: str, limit: int = 500) -> List[int]:
        return [t["digit"] for t in self.tick_buffers.get(market, [])[-limit:]]

    def get_prices(self, market: str, limit: int = 500) -> List[float]:
        return [t["price"] for t in self.tick_buffers.get(market, [])[-limit:]]

    def get_ticks(self, market: str, limit: int = 50) -> List[Dict]:
        return self.tick_buffers.get(market, [])[-limit:]

    def get_tick_count(self, market: str) -> int:
        return len(self.tick_buffers.get(market, []))

    def get_last_digit(self, market: str) -> Optional[int]:
        buf = self.tick_buffers.get(market, [])
        return buf[-1]["digit"] if buf else None

    def get_last_price(self, market: str) -> Optional[float]:
        buf = self.tick_buffers.get(market, [])
        return buf[-1]["price"] if buf else None


deriv_client = DerivWSClient()
