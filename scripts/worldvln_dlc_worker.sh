#!/usr/bin/env bash
set -euo pipefail

UAVEVAL_ROOT="${UAVEVAL_ROOT:-/mnt/petrelfs/youzhongrui/v2/UAVEval}"
AIRBRAIN_ROOT="${AIRBRAIN_ROOT:-/mnt/petrelfs/youzhongrui/v2/AirBrain}"
WORLDVLN_ROOT="${WORLDVLN_ROOT:-/home/liujunli/ceph/liujunli/Ab_ex/WorldVLN}"
WORLDVLN_CKPT_ROOT="${WORLDVLN_CKPT_ROOT:-/home/liujunli/ceph/liujunli/Ab_ex/ckpt/WorldVLN}"
SERVER_PYTHON="${SERVER_PYTHON:-/mnt/petrelfs/youzhongrui/miniconda3/envs/internnav-infer/bin/python}"
EVAL_PYTHON="${EVAL_PYTHON:-/mnt/petrelfs/youzhongrui/miniconda3/envs/qwen/bin/python}"
RUNTIME_SITE="${RUNTIME_SITE:-${UAVEVAL_ROOT}/.runtime/worldvln}"
EVAL_NAME="${EVAL_NAME:-worldvln_airbrain_env_airsim_16}"
REPO_ID="${REPO_ID:-env_airsim_16}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
RESULT_ROOT="${RESULT_ROOT:-${AIRBRAIN_ROOT}/eval_results/${EVAL_NAME}}"
CONFIG="${CONFIG:-${UAVEVAL_ROOT}/configs/jobs/worldvln_airbrain_env_airsim_16.yaml}"
MODEL_GPU="${MODEL_GPU:-1}"
SIM_GPU="${SIM_GPU:-0}"

mkdir -p "${RESULT_ROOT}"
T5_LINK="${RESULT_ROOT}/flan-t5-xl"
ln -sfn "${WORLDVLN_CKPT_ROOT}/t5" "${T5_LINK}"
NATIVE_LOG="${RESULT_ROOT}/worldvln_native.log"
PROXY_LOG="${RESULT_ROOT}/worldvln_proxy.log"

native_pid=""
proxy_pid=""
cleanup() {
  if [[ -n "${proxy_pid}" ]]; then kill "${proxy_pid}" >/dev/null 2>&1 || true; fi
  if [[ -n "${native_pid}" ]]; then kill "${native_pid}" >/dev/null 2>&1 || true; fi
}
trap cleanup EXIT INT TERM

if [[ ! -x "${SERVER_PYTHON}" ]]; then
  echo "missing WorldVLN Python: ${SERVER_PYTHON}" >&2
  exit 1
fi
if [[ ! -d "${RUNTIME_SITE}" ]]; then
  echo "missing preinstalled WorldVLN runtime packages: ${RUNTIME_SITE}" >&2
  exit 1
fi

export DISPLAY="${DISPLAY:-:9}"
if command -v Xvfb >/dev/null 2>&1; then
  Xvfb "${DISPLAY}" -screen 0 1024x768x24 >"${RESULT_ROOT}/xvfb.log" 2>&1 &
fi

echo "[worldvln-worker] GPUs before startup"
nvidia-smi || true

(
  cd "${WORLDVLN_ROOT}"
  # encode_prompt selects the Hugging Face T5 branch by checking whether the
  # configured checkpoint path contains "flan-t5".
  CUDA_VISIBLE_DEVICES="${MODEL_GPU}" \
  PYTHONPATH="${RUNTIME_SITE}:${WORLDVLN_ROOT}/Worldmodel/runtime" \
  PYTHON_BIN="${SERVER_PYTHON}" \
  HOST=127.0.0.1 PORT=8001 ACTION_HEAD_MODE=tsformer_latent \
  INFINITY_CKPT="${WORLDVLN_CKPT_ROOT}/backbone_stage1.pth" \
  STAGE2_LATENT2ACTION_CKPT="${WORLDVLN_CKPT_ROOT}/action_decoder.pt" \
  VAE_PATH="${WORLDVLN_CKPT_ROOT}/vae/model.safetensors" \
  STAGE2_VAE_PATH="${WORLDVLN_CKPT_ROOT}/vae/model.safetensors" \
  T5_PATH="${T5_LINK}" \
  bash infer/run_server.sh
) >"${NATIVE_LOG}" 2>&1 &
native_pid=$!

"${EVAL_PYTHON}" - <<'PY'
import json
import time
import urllib.request

url = "http://127.0.0.1:8001/health"
deadline = time.time() + 900
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            payload = json.load(response)
        if payload.get("infinity_loaded"):
            print("[worldvln-worker] native health:", payload, flush=True)
            break
    except Exception as exc:
        print(f"[worldvln-worker] waiting for native server: {exc}", flush=True)
    time.sleep(10)
else:
    raise TimeoutError("WorldVLN native server did not become healthy in 900 seconds")
PY

CUDA_VISIBLE_DEVICES="${MODEL_GPU}" \
PYTHONPATH="${UAVEVAL_ROOT}/src" \
"${EVAL_PYTHON}" -m uav_eval.server_cli worldvln \
  --upstream-url http://127.0.0.1:8001 \
  --timeout-s 1800 \
  --action-head-mode tsformer_latent \
  --host 127.0.0.1 \
  --port 18104 >"${PROXY_LOG}" 2>&1 &
proxy_pid=$!

"${EVAL_PYTHON}" - <<'PY'
import json
import time
import urllib.request

url = "http://127.0.0.1:18104/health"
deadline = time.time() + 120
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            payload = json.load(response)
        if payload.get("status") == "ok":
            print("[worldvln-worker] proxy health:", payload, flush=True)
            break
    except Exception as exc:
        print(f"[worldvln-worker] waiting for proxy: {exc}", flush=True)
    time.sleep(2)
else:
    raise TimeoutError("WorldVLN proxy did not become healthy in 120 seconds")
PY

cd "${RESULT_ROOT}"
CUDA_VISIBLE_DEVICES="${SIM_GPU}" \
PYTHONPATH="${UAVEVAL_ROOT}/src:${AIRBRAIN_ROOT}" \
"${EVAL_PYTHON}" -m uav_eval run --config "${CONFIG}" \
  --repo-id "${REPO_ID}" \
  --max-samples "${MAX_SAMPLES}" \
  --output-jsonl "${RESULT_ROOT}/eval_results.jsonl" \
  --headless \
  --no-video

"${EVAL_PYTHON}" "${UAVEVAL_ROOT}/scripts/validate_eval_jsonl.py" \
  "${RESULT_ROOT}/eval_results.jsonl" \
  --expected "${MAX_SAMPLES}"
