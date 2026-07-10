#!/usr/bin/env bash
set -euo pipefail

UAVEVAL_ROOT="${UAVEVAL_ROOT:-/mnt/petrelfs/youzhongrui/v2/UAVEval}"
DLC_BIN="${DLC_BIN:-/mnt/petrelfs/youzhongrui/bin/dlc}"
DLC_WORKSPACE_ID="${DLC_WORKSPACE_ID:-279521}"
DLC_RESOURCE_ID="${DLC_RESOURCE_ID:-quota1mtv214y8tu}"
DLC_PRIORITY="${DLC_PRIORITY:-9}"
DLC_OVERSOLD_TYPE="${DLC_OVERSOLD_TYPE:-AcceptQuotaOverSold}"
DLC_IMAGE="${DLC_IMAGE:-pj4090acr-registry-vpc.cn-beijing.cr.aliyuncs.com/pj4090/youzhongrui:vulkan-vnc-0507}"
DLC_DATA_SOURCES="${DLC_DATA_SOURCES:-d-rnpuwbph06wa8yzp8e:v1:/oss,d-3ri6drol6oq8cht5q9:v1:/cpfs/user/youzhongrui/,d-3ri6drol6oq8cht5q9:v1:/mnt/petrelfs/youzhongrui,d-x5hqw561pe26bnavyd:v1:/home/liujunli/ceph}"
DLC_RUN_AS_USER="${DLC_RUN_AS_USER:-${USER:-youzhongrui}}"
EVAL_NAME="${EVAL_NAME:-worldvln_airbrain_env_airsim_16_$(date +%Y%m%d_%H%M%S)}"
JOB_NAME="${JOB_NAME:-env-airsim-16-worldvln-$(date +%Y%m%d-%H%M%S)}"

quote_arg() {
  local value="$1"
  printf "'%s'" "$(printf "%s" "$value" | sed "s/'/'\\''/g")"
}

worker_command="cd $(quote_arg "${UAVEVAL_ROOT}") && EVAL_NAME=$(quote_arg "${EVAL_NAME}") bash $(quote_arg "${UAVEVAL_ROOT}/scripts/worldvln_dlc_worker.sh")"
command="bash -lc $(quote_arg "set -e; if [[ \$(id -u) == 0 ]]; then if ! id -u ${DLC_RUN_AS_USER} >/dev/null 2>&1; then useradd -m -s /bin/bash ${DLC_RUN_AS_USER}; fi; exec su - ${DLC_RUN_AS_USER} -s /bin/bash -c $(quote_arg "${worker_command}"); else exec bash -lc $(quote_arg "${worker_command}"); fi")"

echo "Submitting ${JOB_NAME}"
echo "Evaluation results: /mnt/petrelfs/youzhongrui/v2/AirBrain/eval_results/${EVAL_NAME}"
"${DLC_BIN}" submit pytorchjob \
  --name="${JOB_NAME}" \
  --workspace_id="${DLC_WORKSPACE_ID}" \
  --resource_id="${DLC_RESOURCE_ID}" \
  --priority="${DLC_PRIORITY}" \
  --oversold_type="${DLC_OVERSOLD_TYPE}" \
  --workers=1 \
  --worker_cpu="${DLC_WORKER_CPU:-20}" \
  --worker_gpu="${DLC_WORKER_GPU:-2}" \
  --worker_memory="${DLC_WORKER_MEMORY:-180Gi}" \
  --worker_shared_memory="${DLC_WORKER_SHARED_MEMORY:-32Gi}" \
  --worker_image="${DLC_IMAGE}" \
  --data_sources="${DLC_DATA_SOURCES}" \
  --command="${command}"
