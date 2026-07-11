#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <aerialvla|dualvln>" >&2
  exit 2
fi

MODEL="$1"
UAVEVAL_ROOT="${UAVEVAL_ROOT:-/mnt/petrelfs/youzhongrui/v2/UAVEval}"
AIRBRAIN_ROOT="${AIRBRAIN_ROOT:-/mnt/petrelfs/youzhongrui/v2/AirBrain}"
AB_EX_ROOT="${AB_EX_ROOT:-/home/liujunli/ceph/liujunli/Ab_ex}"
EVAL_PYTHON="${EVAL_PYTHON:-/mnt/petrelfs/youzhongrui/miniconda3/envs/qwen/bin/python}"
EVAL_NAME="${EVAL_NAME:-${MODEL}_airbrain_env_airsim_16}"
REPO_ID="${REPO_ID:-env_airsim_16}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
MODEL_ONLY_SMOKE="${MODEL_ONLY_SMOKE:-false}"
AERIAL_DIAGNOSTIC_ONLY="${AERIAL_DIAGNOSTIC_ONLY:-false}"
RESULT_ROOT="${RESULT_ROOT:-${AIRBRAIN_ROOT}/eval_results/${EVAL_NAME}}"
MODEL_GPU="${MODEL_GPU:-1}"
SIM_GPU="${SIM_GPU:-0}"

case "${MODEL}" in
  aerialvla)
    SERVER_PYTHON="${SERVER_PYTHON:-/mnt/petrelfs/youzhongrui/miniconda3/envs/uavflow/bin/python}"
    SERVER_PORT=18101
    CONFIG="${CONFIG:-${UAVEVAL_ROOT}/configs/models/aerialvla.yaml}"
    server_args=(
      aerialvla
      --repo-root "${AB_EX_ROOT}/AerialVLA"
      --ckpt-dir "${AB_EX_ROOT}/ckpt/AerialVLA"
      --device cuda
      --dtype bfloat16
    )
    ;;
  dualvln)
    SERVER_PYTHON="${SERVER_PYTHON:-/mnt/petrelfs/youzhongrui/miniconda3/envs/internnav-infer/bin/python}"
    SERVER_PORT=18103
    CONFIG="${CONFIG:-${UAVEVAL_ROOT}/configs/models/dualvln.yaml}"
    server_args=(
      dualvln
      --repo-root "${AB_EX_ROOT}/DualVLN"
      --ckpt-dir "${AB_EX_ROOT}/ckpt/DualVLN"
      --device cuda
      --dtype bfloat16
      --predict-steps 32
      --inference-steps 10
      --sample-trajectories 32
    )
    ;;
  *)
    echo "unsupported model: ${MODEL}" >&2
    exit 2
    ;;
esac

mkdir -p "${RESULT_ROOT}"

if [[ "${MODEL}" == "aerialvla" && "${AERIAL_DIAGNOSTIC_ONLY}" == "true" ]]; then
  diagnostic_root="${RESULT_ROOT}/diagnostic"
  mkdir -p "${diagnostic_root}"
  cp "${AB_EX_ROOT}/AerialVLA/src/aerialvla_action_codec.py" "${diagnostic_root}/"
  cp "${AB_EX_ROOT}/AerialVLA/src/train_openfly_lerobot.py" "${diagnostic_root}/"
  {
    echo "[base]"
    find "${AB_EX_ROOT}/ckpt/AerialVLA/openvla-7b" -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
    echo "[lora]"
    find "${AB_EX_ROOT}/ckpt/AerialVLA/lora" -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
  } >"${diagnostic_root}/checkpoint_files.txt"
  for root in "${AB_EX_ROOT}/ckpt/AerialVLA/openvla-7b" "${AB_EX_ROOT}/ckpt/AerialVLA/lora"; do
    label="$(basename "${root}")"
    for name in tokenizer_config.json tokenizer.json tokenizer.model special_tokens_map.json added_tokens.json adapter_config.json config.json; do
      if [[ -f "${root}/${name}" ]]; then
        cp "${root}/${name}" "${diagnostic_root}/${label}_${name}"
      fi
    done
  done
  echo "[model-worker] AerialVLA diagnostic assets: ${diagnostic_root}"
  exit 0
fi

SERVER_LOG="${RESULT_ROOT}/${MODEL}_server.log"
server_pid=""
cleanup() {
  if [[ -n "${server_pid}" ]]; then kill "${server_pid}" >/dev/null 2>&1 || true; fi
}
trap cleanup EXIT INT TERM

export DISPLAY="${DISPLAY:-:9}"
if command -v Xvfb >/dev/null 2>&1; then
  Xvfb "${DISPLAY}" -screen 0 1024x768x24 >"${RESULT_ROOT}/xvfb.log" 2>&1 &
fi

echo "[model-worker] model=${MODEL} repo_id=${REPO_ID} max_samples=${MAX_SAMPLES}"
nvidia-smi || true

CUDA_VISIBLE_DEVICES="${MODEL_GPU}" \
PYTHONPATH="${UAVEVAL_ROOT}/src" \
"${SERVER_PYTHON}" -m uav_eval.server_cli "${server_args[@]}" \
  --host 127.0.0.1 \
  --port "${SERVER_PORT}" >"${SERVER_LOG}" 2>&1 &
server_pid=$!

SERVER_PORT="${SERVER_PORT}" MODEL="${MODEL}" "${EVAL_PYTHON}" - <<'PY'
import json
import os
import time
import urllib.request

url = f"http://127.0.0.1:{os.environ['SERVER_PORT']}/health"
deadline = time.time() + 900
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            payload = json.load(response)
        if payload.get("status") == "ok":
            print(f"[model-worker] {os.environ['MODEL']} health: {payload}", flush=True)
            break
    except Exception as exc:
        print(f"[model-worker] waiting for {os.environ['MODEL']}: {exc}", flush=True)
    time.sleep(10)
else:
    raise TimeoutError(f"{os.environ['MODEL']} did not become healthy in 900 seconds")
PY

if [[ "${MODEL_ONLY_SMOKE}" == "true" ]]; then
  SERVER_PORT="${SERVER_PORT}" MODEL="${MODEL}" "${EVAL_PYTHON}" - <<'PY'
import base64
import io
import json
import os
import urllib.error
import urllib.request

from PIL import Image

buffer = io.BytesIO()
Image.new("RGB", (640, 480), color=(96, 128, 160)).save(buffer, format="PNG")
image = base64.b64encode(buffer.getvalue()).decode("ascii")
base_url = f"http://127.0.0.1:{os.environ['SERVER_PORT']}"
common = {"episode_id": "http-smoke", "instruction": "Fly toward the target building.", "env_name": "env_airsim_16"}
for path, payload in (
    ("/v1/reset", common),
    ("/v1/predict", {**common, "step": 0, "state": [0, 0, 0, 0], "pose": [0, 0, 0, 0], "image_base64": image, "image_format": "png_rgb"}),
):
    request = urllib.request.Request(
        base_url + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{path} returned HTTP {exc.code}: {exc.read().decode()}") from exc
    print(f"[model-worker] {os.environ['MODEL']} {path}: {result}", flush=True)
PY
  exit 0
fi

cd "${RESULT_ROOT}"
CUDA_VISIBLE_DEVICES="${SIM_GPU}" \
PYTHONPATH="${UAVEVAL_ROOT}/src:${AIRBRAIN_ROOT}" \
"${EVAL_PYTHON}" -m uav_eval run \
  --config "${CONFIG}" \
  --repo-id "${REPO_ID}" \
  --max-samples "${MAX_SAMPLES}" \
  --output-jsonl "${RESULT_ROOT}/eval_results.jsonl" \
  --headless \
  --no-video

"${EVAL_PYTHON}" "${UAVEVAL_ROOT}/scripts/validate_eval_jsonl.py" \
  "${RESULT_ROOT}/eval_results.jsonl" \
  --expected "${MAX_SAMPLES}"
