#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <aerialvla|dualvln>" >&2
  exit 2
fi
MODEL="$1"
case "${MODEL}" in aerialvla|dualvln) ;; *) echo "unsupported model: ${MODEL}" >&2; exit 2 ;; esac

UAVEVAL_ROOT="${UAVEVAL_ROOT:-/mnt/petrelfs/youzhongrui/v2/UAVEval}"
DLC_BIN="${DLC_BIN:-/mnt/petrelfs/youzhongrui/bin/dlc}"
DLC_WORKSPACE_ID="${DLC_WORKSPACE_ID:-279521}"
DLC_RESOURCE_ID="${DLC_RESOURCE_ID:-quota1mtv214y8tu}"
DLC_IMAGE="${DLC_IMAGE:-pj4090acr-registry-vpc.cn-beijing.cr.aliyuncs.com/pj4090/youzhongrui:vulkan-vnc-0507}"
DLC_DATA_SOURCES="${DLC_DATA_SOURCES:-d-rnpuwbph06wa8yzp8e:v1:/oss,d-3ri6drol6oq8cht5q9:v1:/cpfs/user/youzhongrui/,d-3ri6drol6oq8cht5q9:v1:/mnt/petrelfs/youzhongrui,d-x5hqw561pe26bnavyd:v1:/home/liujunli/ceph}"
DLC_RUN_AS_USER="${DLC_RUN_AS_USER:-youzhongrui}"
REPO_ID="${REPO_ID:-env_airsim_16}"
REPO_SLUG="${REPO_ID//_/-}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
MODEL_ONLY_SMOKE="${MODEL_ONLY_SMOKE:-false}"
AERIAL_DIAGNOSTIC_ONLY="${AERIAL_DIAGNOSTIC_ONLY:-false}"
EVAL_NAME="${EVAL_NAME:-${MODEL}_airbrain_${REPO_ID}_$(date +%Y%m%d_%H%M%S)}"
JOB_NAME="${JOB_NAME:-${REPO_SLUG}-${MODEL}-$(date +%Y%m%d-%H%M%S)}"

for value in "${MODEL}" "${REPO_ID}" "${EVAL_NAME}" "${DLC_RUN_AS_USER}"; do
  if [[ ! "${value}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "unsafe DLC argument: ${value}" >&2
    exit 2
  fi
done

quote_arg() {
  local value="$1"
  printf "'%s'" "$(printf "%s" "$value" | sed "s/'/'\\''/g")"
}

worker_command="env MODEL_GPU=1 SIM_GPU=0 EVAL_NAME=${EVAL_NAME} REPO_ID=${REPO_ID} MAX_SAMPLES=${MAX_SAMPLES} MODEL_ONLY_SMOKE=${MODEL_ONLY_SMOKE} AERIAL_DIAGNOSTIC_ONLY=${AERIAL_DIAGNOSTIC_ONLY} bash ${UAVEVAL_ROOT}/scripts/model_dlc_worker.sh ${MODEL}"
command="bash -lc $(quote_arg "set -e; cd ${UAVEVAL_ROOT}; if [[ \$(id -u) == 0 ]]; then if ! id -u ${DLC_RUN_AS_USER} >/dev/null 2>&1; then useradd -m -s /bin/bash ${DLC_RUN_AS_USER}; fi; exec runuser -u ${DLC_RUN_AS_USER} -- ${worker_command}; else exec ${worker_command}; fi")"
echo "Submitting ${JOB_NAME}"
echo "Evaluation results: /mnt/petrelfs/youzhongrui/v2/AirBrain/eval_results/${EVAL_NAME}"
"${DLC_BIN}" submit pytorchjob \
  --name="${JOB_NAME}" \
  --workspace_id="${DLC_WORKSPACE_ID}" \
  --resource_id="${DLC_RESOURCE_ID}" \
  --priority="${DLC_PRIORITY:-9}" \
  --oversold_type="${DLC_OVERSOLD_TYPE:-AcceptQuotaOverSold}" \
  --workers=1 \
  --worker_cpu="${DLC_WORKER_CPU:-16}" \
  --worker_gpu="${DLC_WORKER_GPU:-2}" \
  --worker_memory="${DLC_WORKER_MEMORY:-120Gi}" \
  --worker_shared_memory="${DLC_WORKER_SHARED_MEMORY:-32Gi}" \
  --worker_image="${DLC_IMAGE}" \
  --data_sources="${DLC_DATA_SOURCES}" \
  --command="${command}"
