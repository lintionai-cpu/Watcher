#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  Deriv Intelligence Platform — Quick Start
# ─────────────────────────────────────────────────────────────
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND="$SCRIPT_DIR/backend"
VENV="$SCRIPT_DIR/.venv"

echo ""
echo "  ╔══════════════════════════════════════════════╗"
echo "  ║  Deriv Intelligence Platform v2.0            ║"
echo "  ║  AI-Powered Signal Analysis Engine           ║"
echo "  ╚══════════════════════════════════════════════╝"
echo ""

# Create virtualenv if missing
if [ ! -d "$VENV" ]; then
  echo "→ Creating Python virtual environment…"
  python3 -m venv "$VENV"
fi

# Activate
source "$VENV/bin/activate"

# Install / upgrade deps
echo "→ Installing dependencies…"
pip install -q -r "$SCRIPT_DIR/requirements.txt"

# Start backend
echo "→ Starting backend at http://localhost:8000"
echo "→ Frontend at http://localhost:8000/app/"
echo "→ Press Ctrl+C to stop"
echo ""
cd "$BACKEND"
python main.py
