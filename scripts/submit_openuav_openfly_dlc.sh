#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AEROLOOP_ROOT="${AEROLOOP_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
CONFIG="${CONFIG:-${AEROLOOP_ROOT}/configs/jobs/openuav_openfly_env_airsim_16.yaml}"
DLC_BIN="${DLC_BIN:-/mnt/petrelfs/youzhongrui/bin/dlc}"
DLC_WORKSPACE_ID="${DLC_WORKSPACE_ID:-279521}"
DLC_RESOURCE_ID="${DLC_RESOURCE_ID:-quota1mtv214y8tu}"
DLC_IMAGE="${DLC_IMAGE:-pj4090acr-registry-vpc.cn-beijing.cr.aliyuncs.com/pj4090/youzhongrui:vulkan-vnc-0507}"
DLC_DATA_SOURCES="${DLC_DATA_SOURCES:-d-rnpuwbph06wa8yzp8e:v1:/oss,d-3ri6drol6oq8cht5q9:v1:/cpfs/user/youzhongrui/,d-3ri6drol6oq8cht5q9:v1:/mnt/petrelfs/youzhongrui,d-x5hqw561pe26bnavyd:v1:/home/liujunli/ceph}"
DLC_RUN_AS_USER="${DLC_RUN_AS_USER:-youzhongrui}"
DLC_API_HOST="${DLC_API_HOST:-pai-dlc.cn-beijing.aliyuncs.com}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
OPENUAV_STOP_THRESHOLD="${OPENUAV_STOP_THRESHOLD:-0.5}"
EVAL_ENV="${EVAL_ENV:-}"
MODEL_GPU="${MODEL_GPU:-0}"
SIM_GPU="${SIM_GPU:-0}"
EVAL_NAME="${EVAL_NAME:-openuav_openfly_env_airsim_16_$(date +%Y%m%d_%H%M%S)}"
JOB_NAME="${JOB_NAME:-env-airsim-16-openuav-$(date +%Y%m%d-%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-${AEROLOOP_ROOT}/eval_results/dlc/${EVAL_NAME}}"

# Local development shells may route HTTPS through a proxy that closes DLC's
# TLS connection.  Bypass it only for the configured DLC API hostname.
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${DLC_API_HOST}"
export no_proxy="${no_proxy:+${no_proxy},}${DLC_API_HOST}"

for value in "${EVAL_NAME}" "${JOB_NAME}" "${DLC_RUN_AS_USER}"; do
  if [[ ! "${value}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "unsafe DLC argument: ${value}" >&2
    exit 2
  fi
done
if [[ -n "${EVAL_ENV}" && ! "${EVAL_ENV}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "unsafe evaluation environment: ${EVAL_ENV}" >&2
  exit 2
fi

quote_arg() {
  local value="$1"
  printf "'%s'" "$(printf "%s" "$value" | sed "s/'/'\\''/g")"
}

worker_command="env MODEL_GPU=${MODEL_GPU} SIM_GPU=${SIM_GPU} MAX_SAMPLES=${MAX_SAMPLES} OPENUAV_STOP_THRESHOLD=${OPENUAV_STOP_THRESHOLD} EVAL_ENV=${EVAL_ENV} CONFIG=${CONFIG} RESULT_ROOT=${RESULT_ROOT} bash ${AEROLOOP_ROOT}/scripts/openuav_openfly_dlc_worker.sh"
command="bash -lc $(quote_arg "set -e; cd ${AEROLOOP_ROOT}; if [[ \$(id -u) == 0 ]]; then if ! id -u ${DLC_RUN_AS_USER} >/dev/null 2>&1; then useradd -m -s /bin/bash ${DLC_RUN_AS_USER}; fi; exec runuser -u ${DLC_RUN_AS_USER} -- ${worker_command}; else exec ${worker_command}; fi")"

echo "Submitting ${JOB_NAME}"
echo "Evaluation results: ${RESULT_ROOT}"
"${DLC_BIN}" submit pytorchjob \
  --name="${JOB_NAME}" \
  --workspace_id="${DLC_WORKSPACE_ID}" \
  --resource_id="${DLC_RESOURCE_ID}" \
  --priority="${DLC_PRIORITY:-9}" \
  --oversold_type="${DLC_OVERSOLD_TYPE:-AcceptQuotaOverSold}" \
  --workers=1 \
  --worker_cpu="${DLC_WORKER_CPU:-16}" \
  --worker_gpu="${DLC_WORKER_GPU:-1}" \
  --worker_memory="${DLC_WORKER_MEMORY:-120Gi}" \
  --worker_shared_memory="${DLC_WORKER_SHARED_MEMORY:-32Gi}" \
  --worker_image="${DLC_IMAGE}" \
  --data_sources="${DLC_DATA_SOURCES}" \
  --command="${command}"
