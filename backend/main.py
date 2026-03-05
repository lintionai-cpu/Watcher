"""
Deriv Intelligence Platform — FastAPI Backend
"""
import sys, os
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import asyncio, json, logging, time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from data_store    import init_db
from deriv_ws      import deriv_client, MARKETS
from signal_engine import signal_engine
from trade_executor import trade_executor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Deriv Intelligence Platform starting...")
    await init_db()

    # ── Wire deriv_client → signal_engine ─────────────────────────────
    async def on_tick(td):      await signal_engine.broadcast({"type": "tick",              "data": td})
    async def on_status(s):     await signal_engine.broadcast({"type": "connection_status", "data": s})
    async def on_account(a):    await signal_engine.broadcast({"type": "account_info",      "data": a})
    async def on_error(e):      await signal_engine.broadcast({"type": "error",             "data": e})

    deriv_client.on_tick          = on_tick
    deriv_client.on_status_change = on_status
    deriv_client.on_account_info  = on_account
    deriv_client.on_error         = on_error

    # ── Wire deriv_client → trade_executor ────────────────────────────
    trade_executor.set_ws_client(deriv_client)

    deriv_client.on_proposal        = trade_executor.on_proposal_response
    deriv_client.on_buy             = trade_executor.on_buy_response
    deriv_client.on_contract_update = trade_executor.on_contract_update

    # ── Wire trade_executor broadcasts → frontend ──────────────────────
    async def on_trade_event(msg):
        await signal_engine.broadcast(msg)

    trade_executor.on_trade_update = on_trade_event

    # ── Wire signal_engine → trade_executor ───────────────────────────
    signal_engine.set_trade_executor(trade_executor)

    yield

    logger.info("Shutting down...")
    signal_engine.stop()
    await deriv_client.disconnect()


app = FastAPI(title="Deriv Intelligence Platform", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

from fastapi.responses import RedirectResponse

# Try multiple candidate paths — works locally (Windows/Linux) and in Railway/Docker
_FRONTEND_CANDIDATES = [
    os.path.normpath(os.path.join(_BACKEND_DIR, "..", "frontend")),  # relative to backend/
    os.path.join("/app", "frontend"),                                  # Railway: WORKDIR /app
    os.path.join(os.getcwd(), "frontend"),                             # cwd-relative
    os.path.join(os.getcwd(), "..", "frontend"),
]
_FRONTEND = next((p for p in _FRONTEND_CANDIDATES if os.path.isdir(p)), None)

if _FRONTEND:
    logger.info("Frontend found at: %s", _FRONTEND)
    app.mount("/app", StaticFiles(directory=_FRONTEND, html=True), name="frontend")
else:
    logger.warning("Frontend not found. Tried: %s", _FRONTEND_CANDIDATES)

@app.get("/")
async def root():
    return RedirectResponse(url="/app/")


# ── Request models ─────────────────────────────────────────────────────────────
class ConnectRequest(BaseModel):
    api_token: str

class TradeSettingsRequest(BaseModel):
    settings: dict


# ── REST endpoints ─────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}

@app.post("/api/connect")
async def connect(req: ConnectRequest):
    if not req.api_token.strip():
        raise HTTPException(400, "API token required")
    await deriv_client.connect(req.api_token.strip())
    signal_engine.start()
    return {"status": "connecting"}

@app.post("/api/disconnect")
async def disconnect():
    signal_engine.stop()
    trade_executor.settings["enabled"] = False
    await deriv_client.disconnect()
    return {"status": "disconnected"}

@app.get("/api/status")
async def status():
    return {
        "deriv":  {
            "connected": deriv_client.connected,
            "authorized": deriv_client.authorized,
            "reconnect_count": deriv_client.reconnect_count,
            "account": deriv_client.account_info,
        },
        "engine": signal_engine.get_status(),
        "markets": {m: {"name": n,
                        "tick_count": deriv_client.get_tick_count(m),
                        "last_digit": deriv_client.get_last_digit(m),
                        "last_price": deriv_client.get_last_price(m)}
                    for m, n in MARKETS.items()},
    }

@app.get("/api/signals")
async def get_signals(market: Optional[str] = None):
    sigs = signal_engine.get_all_latest_signals()
    if market: sigs = [s for s in sigs if s["market"] == market]
    return {"signals": sigs, "count": len(sigs), "timestamp": time.time()}

@app.get("/api/signals/{market}")
async def get_market_signals(market: str):
    if market not in MARKETS: raise HTTPException(404, f"Unknown market: {market}")
    return {"market": market, "market_name": MARKETS[market],
            "signals": signal_engine.get_live_signals(market),
            "tick_count": deriv_client.get_tick_count(market),
            "last_digit": deriv_client.get_last_digit(market),
            "last_price": deriv_client.get_last_price(market)}

@app.get("/api/ticks/{market}")
async def get_ticks(market: str, limit: int = 50):
    if market not in MARKETS: raise HTTPException(404, f"Unknown market: {market}")
    return {"market": market,
            "ticks":  deriv_client.get_ticks(market, limit),
            "digits": deriv_client.get_digits(market, limit)}

@app.get("/api/analytics/{market}")
async def get_analytics(market: str):
    if market not in MARKETS: raise HTTPException(404, f"Unknown market: {market}")
    return signal_engine.get_digit_analysis(market)

@app.get("/api/markets")
async def get_markets():
    return {m: {"name": n,
                "tick_count": deriv_client.get_tick_count(m),
                "last_digit": deriv_client.get_last_digit(m),
                "last_price": deriv_client.get_last_price(m)}
            for m, n in MARKETS.items()}

# ── Auto-trade REST endpoints ──────────────────────────────────────────────────
@app.get("/api/autotrade/status")
async def autotrade_status():
    return trade_executor.get_status()

@app.post("/api/autotrade/settings")
async def autotrade_settings(req: TradeSettingsRequest):
    trade_executor.update_settings(req.settings)
    return {"ok": True, "settings": trade_executor.settings}

@app.post("/api/autotrade/enable")
async def autotrade_enable():
    trade_executor.settings["enabled"] = True
    await signal_engine.broadcast({"type": "autotrade_status",
                                    "data": {"enabled": True}})
    return {"enabled": True}

@app.post("/api/autotrade/disable")
async def autotrade_disable():
    trade_executor.settings["enabled"] = False
    await signal_engine.broadcast({"type": "autotrade_status",
                                    "data": {"enabled": False}})
    return {"enabled": False}

@app.post("/api/autotrade/reset")
async def autotrade_reset():
    trade_executor.reset_session()
    return {"ok": True}

# ── Backend control ────────────────────────────────────────────────────────────
@app.get("/api/backend/status")
async def backend_status():
    return signal_engine.get_status()

@app.post("/api/backend/start")
async def start_engine():
    signal_engine.start()
    return {"status": "started"}

@app.post("/api/backend/stop")
async def stop_engine():
    signal_engine.stop()
    return {"status": "stopped"}

@app.post("/api/backend/reset")
async def reset_engine():
    signal_engine.stop()
    signal_engine.cache.clear()
    await asyncio.sleep(0.5)
    signal_engine.start()
    return {"status": "reset"}


# ── WebSocket ──────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    signal_engine.register_client(websocket)

    await websocket.send_text(json.dumps({
        "type": "init",
        "data": {
            "connected":    deriv_client.connected,
            "authorized":   deriv_client.authorized,
            "account":      deriv_client.account_info,
            "markets":      {m: {"name": n,
                                 "tick_count": deriv_client.get_tick_count(m),
                                 "last_digit": deriv_client.get_last_digit(m)}
                             for m, n in MARKETS.items()},
            "signals":      signal_engine.get_all_latest_signals(),
            "autotrade":    trade_executor.get_status(),
        },
    }))

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                await _handle_ws(websocket, json.loads(raw))
            except Exception as e:
                logger.warning("WS msg error: %s", e)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning("WS error: %s", e)
    finally:
        signal_engine.unregister_client(websocket)


async def _handle_ws(ws: WebSocket, msg: dict):
    t = msg.get("type")

    if t == "connect":
        token = msg.get("token", "").strip()
        if token:
            await deriv_client.connect(token)
            signal_engine.start()
            await ws.send_text(json.dumps({"type": "ack", "action": "connect"}))

    elif t == "disconnect":
        signal_engine.stop()
        trade_executor.settings["enabled"] = False
        await deriv_client.disconnect()
        await ws.send_text(json.dumps({"type": "ack", "action": "disconnect"}))

    elif t == "autotrade_settings":
        trade_executor.update_settings(msg.get("settings", {}))
        await ws.send_text(json.dumps({"type": "autotrade_status",
                                        "data": trade_executor.get_status()}))

    elif t == "autotrade_enable":
        trade_executor.settings["enabled"] = True
        await signal_engine.broadcast({"type": "autotrade_status",
                                        "data": {"enabled": True,
                                                 **trade_executor.session.to_dict()}})

    elif t == "autotrade_disable":
        trade_executor.settings["enabled"] = False
        await signal_engine.broadcast({"type": "autotrade_status",
                                        "data": {"enabled": False,
                                                 **trade_executor.session.to_dict()}})

    elif t == "autotrade_reset":
        trade_executor.reset_session()
        await ws.send_text(json.dumps({"type": "autotrade_status",
                                        "data": trade_executor.get_status()}))

    elif t == "get_signals":
        await ws.send_text(json.dumps({
            "type": "signals_response",
            "data": signal_engine.get_live_signals(msg.get("market"))
        }))

    elif t == "get_analytics":
        market = msg.get("market")
        if market and market in MARKETS:
            await ws.send_text(json.dumps({
                "type": "analytics_response",
                "data": signal_engine.get_digit_analysis(market)
            }))

    elif t == "ping":
        await ws.send_text(json.dumps({"type": "pong", "ts": time.time()}))


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))   # Railway injects $PORT
    uvicorn.run("main:app", host="0.0.0.0", port=port,
                reload=False, workers=1, log_level="info")
