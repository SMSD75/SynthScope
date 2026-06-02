#!/bin/bash
# Launch the AGID interface backend (FastAPI + uvicorn) from the AGID conda env.
#
#   bash interface/run_backend.sh                 # serve on 0.0.0.0:8000
#   AGID_PORT=9000 bash interface/run_backend.sh   # custom port
#
# The backend serves the built Vite frontend (interface/frontend/dist) if present, else the
# no-build standalone page (interface/static/index.html). Models load lazily on first request.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PORT="${AGID_PORT:-8000}"
HOST="${AGID_HOST:-0.0.0.0}"
CONDA="${CONDA_EXE:-/home/ssalehi/miniconda3/bin/conda}"

# Caches default to a node-local location (see backend/config.py). Override as needed:
# export AGID_CKPT_DIR=/path/to/checkpoints

echo "Serving AGID interface on http://${HOST}:${PORT}  (Ctrl-C to stop)"
exec "$CONDA" run --no-capture-output -n AGID \
  uvicorn interface.backend.app:app --host "$HOST" --port "$PORT" "$@"
