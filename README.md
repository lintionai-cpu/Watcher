# Deriv Intelligence Platform v2.0

A professional 24/7 AI-powered market analysis and signal generation platform for Deriv volatility indices.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│  Browser (Frontend — index.html)            │
│  • Multi-page SPA (Milky White + Gold UI)   │
│  • Real-time WebSocket signal display       │
│  • Dashboard, Signals, Analytics, Monitor  │
└──────────────┬──────────────────────────────┘
               │ WebSocket + REST
               │ ws://localhost:8000/ws
               │ http://localhost:8000/api
┌──────────────▼──────────────────────────────┐
│  FastAPI Backend (main.py)                  │
│  ┌────────────────────────────────────────┐ │
│  │  Signal Engine — runs 24/7             │ │
│  │  ┌──────────────────────────────────┐  │ │
│  │  │  Analysis Engine per market      │  │ │
│  │  │  • Markov Model                  │  │ │
│  │  │  • HMM (3 regimes)               │  │ │
│  │  │  • Autocorrelation + FFT         │  │ │
│  │  │  • ARIMA time-series             │  │ │
│  │  │  • K-Means clustering            │  │ │
│  │  │  • SVM + Random Forest           │  │ │
│  │  │  • Lightweight LSTM (numpy)      │  │ │
│  │  │  • Apriori pattern mining        │  │ │
│  │  │  • Genetic Algorithm rules       │  │ │
│  │  │  • Shannon Entropy + Chi-Square  │  │ │
│  │  │  • Q-Learning ensemble weights   │  │ │
│  │  └──────────────────────────────────┘  │ │
│  └────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────┐ │
│  │  Deriv WebSocket Client                │ │
│  │  • Auto-reconnect + heartbeat          │ │
│  │  • 10 markets subscribed               │ │
│  │  • Tick history on connect             │ │
│  └────────────────────────────────────────┘ │
│  SQLite Database (data/deriv_intel.db)      │
└──────────────┬──────────────────────────────┘
               │ WSS
               │ wss://ws.derivws.com/websockets/v3
┌──────────────▼──────────────────────────────┐
│  Deriv Servers                              │
│  • R_10, R_25, R_50, R_75, R_100            │
│  • 1HZ10V, 1HZ25V, 1HZ50V, 1HZ75V, 1HZ100V│
└─────────────────────────────────────────────┘
```

---

## Strategy Categories

| Strategy | Barrier Over | Barrier Under | Probability |
|----------|-------------|---------------|-------------|
| Over 2 / Under 7 | DIGIT > 2 | DIGIT < 7 | ~70% |
| Under 5 / Over 5 | DIGIT > 4 | DIGIT < 5 | ~50% |
| Over 1 / Under 8 | DIGIT > 1 | DIGIT < 8 | ~80% |
| Under 3 / Over 5 | DIGIT > 5 | DIGIT < 3 | ~40% |

### Signal Fields
- **Direction**: OVER or UNDER
- **Barrier**: The digit threshold
- **Confidence**: 0–100% ensemble confidence
- **Valid Window**: How long (seconds) the signal remains actionable
- **Duration Ticks**: Recommended contract duration in ticks

---

## Setup

### Prerequisites
- Python 3.10+
- pip

### Install
```bash
cd backend
pip install -r ../requirements.txt
```

### Run Backend
```bash
cd backend
python main.py
# Server starts at http://localhost:8000
```

### Open Frontend
Open `frontend/index.html` in your browser **or** go to:
```
http://localhost:8000/app/
```

---

## 24/7 Deployment (systemd)

```bash
sudo nano /etc/systemd/system/deriv-intel.service
```

```ini
[Unit]
Description=Deriv Intelligence Platform
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/deriv_platform/backend
ExecStart=/usr/bin/python3 main.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable deriv-intel
sudo systemctl start deriv-intel
sudo systemctl status deriv-intel
```

---

## 24/7 Deployment (PM2 — Node.js process manager)
```bash
npm install -g pm2
cd backend
pm2 start "python main.py" --name deriv-intel
pm2 save
pm2 startup
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/connect | Connect to Deriv with token |
| POST | /api/disconnect | Disconnect |
| GET | /api/status | Full system status |
| GET | /api/signals | All active signals |
| GET | /api/signals/{market} | Signals for one market |
| GET | /api/ticks/{market} | Recent ticks |
| GET | /api/analytics/{market} | Digit analysis |
| GET | /api/markets | Market summary |
| POST | /api/backend/start | Start engine |
| POST | /api/backend/stop | Stop engine |
| POST | /api/backend/reset | Reset & restart |
| WS | /ws | Real-time WebSocket |

---

## Frontend Pages

1. **Dashboard** — Overview, connection, market cards, recent signals
2. **Signals Feed** — All live signals with full detail, filterable by market
3. **Strategy Pages** — Dedicated page per strategy category showing signals for all markets
4. **Analytics** — Digit frequency, distribution, entropy, zone analysis with charts
5. **Backend Monitor** — Control engine, monitor market data status
6. **Settings** — API token, backend URL configuration

---

## Notes

- **Minimum 60 ticks** required per market before analysis begins
- Signals auto-expire after their **Valid Window** elapses
- **Q-Learning** adjusts model weights over time based on historical accuracy
- The backend runs independently of the frontend — you can close the browser and signals keep generating
- All tick data and signals are stored in SQLite at `data/deriv_intel.db`
