"""
Signal Engine — analysis loop, signal cache, WebSocket broadcast,
and auto-trade trigger.
"""
import asyncio, json, logging, time
from typing import Dict, List, Optional, Set, Any

from analysis_engine import get_analyzer, STRATEGY_CATEGORIES
from deriv_ws        import deriv_client, MARKETS

logger            = logging.getLogger(__name__)
ANALYSIS_INTERVAL = 3.0
MIN_TICKS_REQUIRED = 60


class SignalCache:
    def __init__(self):
        self._cache: Dict[str, Dict] = {}

    def put(self, signal: Dict):
        self._cache[f"{signal['market']}:{signal['category']}"] = signal

    def get(self, market: str, category: str) -> Optional[Dict]:
        sig = self._cache.get(f"{market}:{category}")
        if sig and (time.time() - sig["timestamp"]) <= sig["valid_window"]:
            return sig
        return None

    def get_all(self) -> List[Dict]:
        now = time.time()
        return [s for s in self._cache.values()
                if (now - s["timestamp"]) <= s["valid_window"]]

    def get_all_latest(self) -> List[Dict]:
        return list(self._cache.values())

    def clear(self):
        self._cache.clear()


class SignalEngine:
    def __init__(self):
        self.cache            = SignalCache()
        self._running         = False
        self._task: Optional[asyncio.Task] = None
        self._ws_clients: Set[Any] = set()
        self._analysis_stats: Dict[str, Dict] = {}
        self._signal_counts:  Dict[str, int]  = {}
        self._trade_executor  = None          # injected from main.py

    def set_trade_executor(self, executor) -> None:
        self._trade_executor = executor

    # ── Lifecycle ────────────────────────────────────────────────────────
    def start(self):
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._analysis_loop())
            logger.info("Signal engine started")

    def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
        logger.info("Signal engine stopped")

    # ── WebSocket clients ────────────────────────────────────────────────
    def register_client(self, ws):   self._ws_clients.add(ws)
    def unregister_client(self, ws): self._ws_clients.discard(ws)

    async def broadcast(self, message: Dict):
        if not self._ws_clients:
            return
        data = json.dumps(message)
        dead = set()
        for client in self._ws_clients:
            try:
                await client.send_text(data)
            except Exception:
                dead.add(client)
        for c in dead:
            self._ws_clients.discard(c)

    # ── Main loop ────────────────────────────────────────────────────────
    async def _analysis_loop(self):
        logger.info("Analysis loop running")
        while self._running:
            try:
                for market in MARKETS:
                    await self._analyze_market(market)
                    await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Analysis loop error: %s", e)
            await asyncio.sleep(ANALYSIS_INTERVAL)

    async def _analyze_market(self, market: str):
        digits = deriv_client.get_digits(market, 500)
        if len(digits) < MIN_TICKS_REQUIRED:
            return

        analyzer = get_analyzer(market)
        signals  = analyzer.generate_all_signals(digits)

        stats = {
            "market":             market,
            "tick_count":         len(digits),
            "last_digit":         digits[-1] if digits else None,
            "last_price":         deriv_client.get_last_price(market),
            "signals_generated":  len(signals),
            "analyzed_at":        time.time(),
        }
        self._analysis_stats[market] = stats

        for sig in signals:
            key = f"{market}:{sig['category']}"
            self._signal_counts[key] = self._signal_counts.get(key, 0) + 1
            self.cache.put(sig)

            # Broadcast signal to frontend
            await self.broadcast({"type": "signal", "data": sig})

            # Fire auto-trade with zero delay
            if self._trade_executor:
                asyncio.create_task(self._trade_executor.on_signal(sig))

        await self.broadcast({"type": "market_update", "data": stats})

    # ── Data access ──────────────────────────────────────────────────────
    def get_live_signals(self, market: Optional[str] = None) -> List[Dict]:
        if market:
            return [s for cat in STRATEGY_CATEGORIES
                    if (s := self.cache.get(market, cat))]
        return self.cache.get_all()

    def get_all_latest_signals(self) -> List[Dict]:
        return self.cache.get_all_latest()

    def get_market_stats(self) -> Dict:
        return self._analysis_stats.copy()

    def get_status(self) -> Dict:
        return {
            "running":              self._running,
            "connected":            deriv_client.connected,
            "authorized":           deriv_client.authorized,
            "markets_tracked":      len(MARKETS),
            "markets_with_data":    sum(1 for m in MARKETS
                                        if deriv_client.get_tick_count(m) >= MIN_TICKS_REQUIRED),
            "total_signals_cache":  len(self.cache.get_all_latest()),
            "analysis_stats":       self._analysis_stats,
        }

    def get_digit_analysis(self, market: str) -> Dict:
        import numpy as np
        digits = deriv_client.get_digits(market, 500)
        if not digits:
            return {}
        arr   = np.array(digits, dtype=float)
        freq  = [int(np.sum(arr == i)) for i in range(10)]
        total = len(arr)
        return {
            "market":            market,
            "market_name":       MARKETS.get(market, market),
            "tick_count":        total,
            "last_20_digits":    digits[-20:],
            "last_price":        deriv_client.get_last_price(market),
            "digit_frequency":   {str(i): freq[i] for i in range(10)},
            "digit_percentage":  {str(i): round(freq[i] / total * 100, 2) for i in range(10)},
            "mean_digit":        round(float(arr.mean()), 3),
            "std_digit":         round(float(arr.std()),  3),
            "low_zone_pct":      round(float(np.sum(arr < 3) / total * 100), 2),
            "mid_zone_pct":      round(float(np.sum((arr >= 3) & (arr <= 6)) / total * 100), 2),
            "high_zone_pct":     round(float(np.sum(arr > 6) / total * 100), 2),
            "last_streak":       self._calc_streak(digits),
            "volatility_score":  round(float(arr[-50:].std()) if len(arr) >= 50 else 0, 3),
        }

    def _calc_streak(self, digits):
        if not digits: return {}
        last, count = digits[-1], 0
        for d in reversed(digits):
            if d == last: count += 1
            else: break
        return {"digit": last, "count": count}


signal_engine = SignalEngine()
