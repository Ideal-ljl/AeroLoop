#!/usr/bin/env bash
set -euo pipefail

AB_EX_ROOT="${AB_EX_ROOT:-${HOME}/uav-models}"
PYTHON_BIN="${PYTHON_BIN:-python}"
exec "${PYTHON_BIN}" -m aeroloop.server_cli openuav \
  --repo-root "${OPENUAV_ROOT:-${AB_EX_ROOT}/OpenUAV}" \
  --ckpt-dir "${OPENUAV_CKPT:-${AB_EX_ROOT}/OpenUAV/ckpt}" \
  --traj-model-path "${OPENUAV_TRAJ_CKPT:-${AB_EX_ROOT}/OpenUAV/ckpt/model_2.pth}" \
  --device "${DEVICE:-cuda}" \
  --stop-norm-threshold "${STOP_THRESHOLD:-0.5}" \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-18102}"
