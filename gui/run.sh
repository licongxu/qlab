#!/usr/bin/env bash
# Start qlab GUI - backend + frontend dev servers
set -e
cd "$(dirname "$0")/.."

BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

echo "=== qlab GUI ==="
echo "Backend:  http://localhost:$BACKEND_PORT"
echo "Frontend: http://localhost:$FRONTEND_PORT"
echo ""

# Start backend
echo "Starting FastAPI backend on port $BACKEND_PORT..."
python -m uvicorn gui.backend.main:app --host 0.0.0.0 --port "$BACKEND_PORT" --reload &
BACKEND_PID=$!

# Start frontend dev server
echo "Starting Vite dev server on port $FRONTEND_PORT..."
cd gui/frontend
VITE_PORT=$FRONTEND_PORT npx vite --port "$FRONTEND_PORT" --host 0.0.0.0 &
FRONTEND_PID=$!

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
echo ""
echo "Press Ctrl+C to stop both servers."
wait
