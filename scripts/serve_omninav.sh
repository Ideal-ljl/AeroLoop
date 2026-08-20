#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AEROLOOP_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${AEROLOOP_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

AB_EX_ROOT="${AB_EX_ROOT:-/mnt/petrelfs/youzhongrui/v2/Ab_ex}"
DEFAULT_PYTHON="/mnt/petrelfs/youzhongrui/miniconda3/envs/internnav-infer/bin/python"
if [[ ! -x "${DEFAULT_PYTHON}" ]]; then
  DEFAULT_PYTHON="python"
fi
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"
exec "${PYTHON_BIN}" -m aeroloop.server_cli omninav \
  --repo-root "${OMNINAV_ROOT:-${AB_EX_ROOT}/OmniNav}" \
  --ckpt-dir "${OMNINAV_CKPT:-${AB_EX_ROOT}/OmniNav/checkpoint-19805}" \
  --device "${DEVICE:-cuda}" \
  --dtype "${DTYPE:-bfloat16}" \
  --attn-impl "${ATTN_IMPL:-sdpa}" \
  --history-size "${HISTORY_SIZE:-5}" \
  --current-long-edge "${CURRENT_LONG_EDGE:-640}" \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-18105}"
