#!/usr/bin/env bash
# Azure App Service startup script for Streamlit UI + FastAPI backend.
set -euo pipefail

# Oryx can place the built application under APP_PATH rather than wwwroot.
APP_DIR="${APP_PATH:-$(dirname "$0")}"
cd "$APP_DIR"

# App Service routes public traffic to the configured PORT. The provided logs
# show PORT=3001, so Streamlit listens on that port by default.
WEB_PORT="${PORT:-8000}"
API_PORT="${BACKEND_PORT:-3002}"

# Streamlit makes server-side HTTP requests to the internal FastAPI service.
export STREAMLIT_API_BASE="${STREAMLIT_API_BASE:-http://127.0.0.1:${API_PORT}}"

echo "Starting FastAPI backend on 127.0.0.1:${API_PORT}"
python -m uvicorn main:app --host 127.0.0.1 --port "${API_PORT}" &
API_PID=$!

cleanup() {
  echo "Stopping FastAPI backend..."
  kill "${API_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Starting Streamlit on 0.0.0.0:${WEB_PORT}"
python -m streamlit run streamlit_app.py \
  --server.address 0.0.0.0 \
  --server.port "${WEB_PORT}" \
  --server.headless true
