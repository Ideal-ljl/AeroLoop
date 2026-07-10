#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <aerialvla|dualvln|worldvln>" >&2
  exit 2
fi

MODEL="$1"
case "${MODEL}" in aerialvla|dualvln|worldvln) ;; *) echo "unsupported model: ${MODEL}" >&2; exit 2 ;; esac

ROOT="${UAVEVAL_ROOT:-/mnt/petrelfs/youzhongrui/v2/UAVEval}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
EVAL_TAG="${EVAL_TAG:-$(date +%Y%m%d_%H%M%S)}"
REPO_IDS=(
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

for repo_id in "${REPO_IDS[@]}"; do
  eval_name="${MODEL}_airbrain_${repo_id}_${EVAL_TAG}"
  job_name="${repo_id//_/-}-${MODEL}-${EVAL_TAG//_/-}"
  if [[ "${MODEL}" == "worldvln" ]]; then
    REPO_ID="${repo_id}" MAX_SAMPLES="${MAX_SAMPLES}" EVAL_NAME="${eval_name}" JOB_NAME="${job_name}" \
      "${ROOT}/scripts/submit_worldvln_dlc.sh"
  else
    REPO_ID="${repo_id}" MAX_SAMPLES="${MAX_SAMPLES}" EVAL_NAME="${eval_name}" JOB_NAME="${job_name}" \
      "${ROOT}/scripts/submit_model_dlc.sh" "${MODEL}"
  fi
done
