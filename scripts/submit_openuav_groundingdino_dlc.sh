#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AEROLOOP_ROOT="${AEROLOOP_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
export CONFIG="${CONFIG:-${AEROLOOP_ROOT}/configs/jobs/openuav_groundingdino_openfly_env_airsim_16.yaml}"
export EVAL_NAME="${EVAL_NAME:-openuav_groundingdino_env_airsim_16_$(date +%Y%m%d_%H%M%S)}"
export JOB_NAME="${JOB_NAME:-env-airsim-16-openuav-dino-$(date +%Y%m%d-%H%M%S)}"

exec "${AEROLOOP_ROOT}/scripts/submit_openuav_openfly_dlc.sh"
