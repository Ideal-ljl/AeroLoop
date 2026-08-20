#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AEROLOOP_ROOT="${AEROLOOP_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
CONFIG="${CONFIG:-${AEROLOOP_ROOT}/configs/jobs/openuav_l1_oracle_openfly_env_airsim_16.yaml}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
stamp="${EVAL_STAMP:-$(date +%Y%m%d_%H%M%S)}"

default_envs=(
  env_airsim_16
  env_airsim_18
  env_airsim_26
  env_airsim_sh
  env_gs_ecust
  env_gs_nwpu01
  env_gs_nwpu02
  env_gs_sjtu01
  env_gs_sjtu02
  env_ue_bigcity
  env_ue_smallcity
)
if [[ -n "${EVAL_ENVS:-}" ]]; then
  read -r -a environments <<< "${EVAL_ENVS//,/ }"
else
  environments=("${default_envs[@]}")
fi

echo "Submitting ${#environments[@]} OpenUAV L1-oracle evaluations (${MAX_SAMPLES} episodes per environment)"
for env_name in "${environments[@]}"; do
  safe_env="${env_name//_/-}"
  EVAL_ENV="${env_name}" \
  CONFIG="${CONFIG}" \
  MAX_SAMPLES="${MAX_SAMPLES}" \
  OPENUAV_STOP_THRESHOLD=0.0 \
  EVAL_NAME="openuav_l1_oracle_${env_name}_${stamp}" \
  JOB_NAME="${safe_env}-openuav-l1-oracle-${stamp}" \
    bash "${AEROLOOP_ROOT}/scripts/submit_openuav_openfly_dlc.sh"
done
