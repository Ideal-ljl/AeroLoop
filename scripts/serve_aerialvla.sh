#!/usr/bin/env bash
set -euo pipefail

AB_EX_ROOT="${AB_EX_ROOT:-${HOME}/uav-models}"
PYTHON_BIN="${PYTHON_BIN:-python}"
exec "${PYTHON_BIN}" -m aeroloop.server_cli aerialvla \
  --repo-root "${AERIALVLA_ROOT:-${AB_EX_ROOT}/AerialVLA}" \
  --ckpt-dir "${AERIALVLA_CKPT:-${AB_EX_ROOT}/ckpt/AerialVLA}" \
  --device "${DEVICE:-cuda}" \
  --dtype "${DTYPE:-bfloat16}" \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-18101}"
