#!/usr/bin/env bash
set -euo pipefail

AB_EX_ROOT="${AB_EX_ROOT:-${HOME}/uav-models}"
PYTHON_BIN="${PYTHON_BIN:-python}"
exec "${PYTHON_BIN}" -m aeroloop.server_cli dualvln \
  --repo-root "${DUALVLN_ROOT:-${AB_EX_ROOT}/DualVLN}" \
  --ckpt-dir "${DUALVLN_CKPT:-${AB_EX_ROOT}/ckpt/DualVLN}" \
  --device "${DEVICE:-cuda}" \
  --dtype "${DTYPE:-bfloat16}" \
  --predict-steps "${PREDICT_STEPS:-32}" \
  --inference-steps "${DENOISE_STEPS:-10}" \
  --sample-trajectories "${SAMPLE_TRAJECTORIES:-32}" \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-18103}"
