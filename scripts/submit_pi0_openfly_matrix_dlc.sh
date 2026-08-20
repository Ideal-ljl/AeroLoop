#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AEROLOOP_ROOT="${AEROLOOP_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
EVAL_TAG="${EVAL_TAG:-$(date +%Y%m%d_%H%M%S)}"
PI0_CHECKPOINT="${PI0_CHECKPOINT:-/mnt/petrelfs/youzhongrui/v2/Ab_ex/checkpoints/pi0-openfly-16g-bs64-v2-0723/030000/pretrained_model}"
PI0_HISTORY_STEPS="${PI0_HISTORY_STEPS:-1}"
MANIFEST="${MANIFEST:-${AEROLOOP_ROOT}/eval_results/dlc/pi0_openfly_matrix_${EVAL_TAG}.tsv}"
ENVIRONMENTS=(
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
  read -r -a ENVIRONMENTS <<<"${EVAL_ENVS//,/ }"
fi

mkdir -p "$(dirname "${MANIFEST}")"
touch "${MANIFEST}"

for eval_env in "${ENVIRONMENTS[@]}"; do
  eval_name="pi0_openfly_${eval_env}_${EVAL_TAG}"
  job_name="${eval_env//_/-}-pi0-${EVAL_TAG//_/-}"
  result_root="${AEROLOOP_ROOT}/eval_results/dlc/${eval_name}"
  output="$(
    EVAL_ENV="${eval_env}" \
    MAX_SAMPLES="${MAX_SAMPLES}" \
    EVAL_NAME="${eval_name}" \
    JOB_NAME="${job_name}" \
    RESULT_ROOT="${result_root}" \
    PI0_CHECKPOINT="${PI0_CHECKPOINT}" \
    PI0_HISTORY_STEPS="${PI0_HISTORY_STEPS}" \
      "${AEROLOOP_ROOT}/scripts/submit_pi0_openfly_dlc.sh"
  )"
  printf '%s\n' "${output}"
  job_id="$(printf '%s\n' "${output}" | sed -nE 's/.*(dlc[a-z0-9]+).*/\1/p' | tail -n 1)"
  if [[ -z "${job_id}" ]]; then
    echo "failed to parse DLC job id for ${eval_env}" >&2
    exit 1
  fi
  printf '%s\t%s\t%s\t%s\n' "${eval_env}" "${job_id}" "${job_name}" "${result_root}" >>"${MANIFEST}"
done

echo "Matrix manifest: ${MANIFEST}"
