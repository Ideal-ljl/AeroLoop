#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AEROLOOP_ROOT="${AEROLOOP_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
stamp="$(date +%Y%m%d_%H%M%S)"
export CONFIG="${CONFIG:-${AEROLOOP_ROOT}/configs/jobs/openuav_l1_oracle_openfly_env_airsim_16.yaml}"
export EVAL_NAME="${EVAL_NAME:-openuav_l1_oracle_env_airsim_16_${stamp}}"
export JOB_NAME="${JOB_NAME:-env-airsim-16-openuav-l1-oracle-${stamp}}"
# Disable the model's learned stop head for this GT-distance-terminated run.
export OPENUAV_STOP_THRESHOLD="${OPENUAV_STOP_THRESHOLD:-0.0}"

exec "${AEROLOOP_ROOT}/scripts/submit_openuav_openfly_dlc.sh"
