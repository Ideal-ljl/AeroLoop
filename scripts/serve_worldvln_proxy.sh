#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
exec "${PYTHON_BIN}" -m uav_eval.server_cli worldvln \
  --upstream-url "${WORLDVLN_UPSTREAM_URL:-http://127.0.0.1:8001}" \
  --timeout-s "${TIMEOUT_S:-600}" \
  --action-head-mode "${ACTION_HEAD_MODE:-tsformer_latent}" \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-18104}"
