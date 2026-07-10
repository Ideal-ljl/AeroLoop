#!/usr/bin/env bash
set -euo pipefail

AB_EX_ROOT="${AB_EX_ROOT:-/home/liujunli/ceph/liujunli/Ab_ex}"
PYTHON_BIN="${PYTHON_BIN:-python}"
exec "${PYTHON_BIN}" -m uav_eval.server_cli openuav \
  --repo-root "${OPENUAV_ROOT:-${AB_EX_ROOT}/OpenUAV}" \
  --ckpt-dir "${OPENUAV_CKPT:-${AB_EX_ROOT}/ckpt/OpenUAV}" \
  --device "${DEVICE:-cuda}" \
  --stop-norm-threshold "${STOP_THRESHOLD:-0.5}" \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-18102}"
