#!/usr/bin/env bash
# Start qlab GUI - production mode (serves frontend from FastAPI)
set -e
cd "$(dirname "$0")/.."

PORT="${PORT:-8000}"

# Build frontend if dist doesn't exist
if [ ! -d "gui/frontend/dist" ]; then
    echo "Building frontend..."
    cd gui/frontend && npm run build && cd ../..
fi

echo "=== qlab GUI (production) ==="
echo "Open http://localhost:$PORT"
echo ""

python -m uvicorn gui.backend.main:app --host 0.0.0.0 --port "$PORT"
