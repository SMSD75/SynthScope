#!/bin/bash
# Launch the SynthScope interface backend (FastAPI + uvicorn) from the SynthScope conda env.
#
#   bash interface/run_backend.sh                 # serve on 0.0.0.0:8000
#   SYNTHSCOPE_PORT=9000 bash interface/run_backend.sh   # custom port
#
# The backend serves the built Vite frontend (interface/frontend/dist) if present, else the
# no-build standalone page (interface/static/index.html). Models load lazily on first request.
set -euo pipefail

# Old AGID_* variables still work: map each to its SYNTHSCOPE_* name unless that is already set.
for _v in $(compgen -e | grep '^AGID_' || true); do _n="SYNTHSCOPE_${_v#AGID_}"; [ -n "${!_n:-}" ] || export "$_n=${!_v}"; done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PORT="${SYNTHSCOPE_PORT:-8000}"
HOST="${SYNTHSCOPE_HOST:-0.0.0.0}"
CONDA="${CONDA_EXE:-/home/ssalehi/miniconda3/bin/conda}"

# Caches default to a node-local location (see backend/config.py). Override as needed:
# export SYNTHSCOPE_CKPT_DIR=/path/to/checkpoints

echo "Serving SynthScope interface on http://${HOST}:${PORT}  (Ctrl-C to stop)"
exec "$CONDA" run --no-capture-output -n SynthScope \
  uvicorn interface.backend.app:app --host "$HOST" --port "$PORT" "$@"
