#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AEROLOOP_ROOT="${AEROLOOP_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
AB_EX_ROOT="${AB_EX_ROOT:-/mnt/petrelfs/youzhongrui/v2/Ab_ex}"
EVAL_PYTHON="${EVAL_PYTHON:-/mnt/petrelfs/youzhongrui/miniconda3/envs/qwen/bin/python}"
SERVER_PYTHON="${SERVER_PYTHON:-/mnt/petrelfs/youzhongrui/miniconda3/envs/openuav-test/bin/python}"
CONFIG="${CONFIG:-${AEROLOOP_ROOT}/configs/jobs/openuav_openfly_env_airsim_16.yaml}"
RESULT_ROOT="${RESULT_ROOT:?RESULT_ROOT is required}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
SERVER_PORT="${SERVER_PORT:-18102}"
MODEL_GPU="${MODEL_GPU:-1}"
SIM_GPU="${SIM_GPU:-0}"
EVAL_ENV="${EVAL_ENV:-}"

mkdir -p "${RESULT_ROOT}"
RUNTIME_CONFIG="${RESULT_ROOT}/openuav_eval_config.json"
export AEROLOOP_ROOT CONFIG RESULT_ROOT MAX_SAMPLES SERVER_PORT RUNTIME_CONFIG EVAL_ENV
PYTHONPATH="${AEROLOOP_ROOT}/src" "${EVAL_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

from aeroloop.config import load_config

config = load_config(os.environ["CONFIG"])
result_root = Path(os.environ["RESULT_ROOT"])
port = int(os.environ["SERVER_PORT"])
eval_env = os.environ.get("EVAL_ENV", "").strip()
runtime_root = Path(os.environ["AEROLOOP_ROOT"]) / ".runtime" / "envs"
if eval_env:
    config["benchmark"]["kwargs"]["include_envs"] = eval_env
    common_camera = {
        "name": "front",
        "width": 896,
        "height": 896,
        "fov": 90,
    }
    if eval_env.startswith("env_airsim_"):
        config["simulator"] = {
            "type": "airsim",
            "kwargs": {
                "env_root": str(runtime_root / "airsim"),
                "launch_script": "LinuxNoEditor/start.sh",
                "host": "127.0.0.1",
                "port": 41451,
                "cameras": [common_camera],
                "camera_names": {"front": "front_custom"},
                "vehicle_name": "drone_1",
                "configure_camera_poses": False,
                "strict_camera_size": True,
                "capture_depth": False,
                "position_sign": [1, -1, -1],
                "yaw_sign": -1,
                "channel_order": "bgr",
                "ignore_collision": False,
            },
        }
    elif eval_env.startswith("env_gs_"):
        config["simulator"] = {
            "type": "gs_airsim",
            "kwargs": {
                "env_root": str(runtime_root / "gs_airsim"),
                "launch_script": "gs.sh",
                "host": "127.0.0.1",
                "port": 41451,
                "cameras": ["front"],
                "camera_names": {"front": "front_custom"},
                "configure_camera_poses": False,
                "strict_camera_size": False,
                "capture_depth": False,
                "ignore_collision": False,
            },
        }
    elif eval_env.startswith("env_ue_"):
        config["simulator"] = {
            "type": "unrealcv",
            "kwargs": {
                "env_root": str(runtime_root / "ue"),
                "launch_script": "CitySample.sh",
                "host": "127.0.0.1",
                "port": 9000,
                "update_ini_port": True,
                "spawn_cameras": True,
                "cameras": [common_camera],
                "camera_ids": {"front": 1},
                "collision_query": None,
            },
        }
    else:
        raise ValueError(f"unsupported OpenFly evaluation environment: {eval_env!r}")
config["benchmark"]["kwargs"]["limit"] = int(os.environ["MAX_SAMPLES"])
policy_config = config["policy"]
while policy_config.get("type") != "http":
    policy_config = policy_config.get("kwargs", {}).get("base")
    if not isinstance(policy_config, dict):
        raise ValueError("OpenUAV DLC policy stack must contain a nested HTTP base policy")
policy_kwargs = policy_config["kwargs"]
policy_kwargs.update(
    {
        "url": f"http://127.0.0.1:{port}/v1/predict",
        "reset_url": f"http://127.0.0.1:{port}/v1/reset",
    }
)
config["simulator"]["kwargs"]["log_path"] = str(result_root / "simulator.log")
config["output"]["jsonl"] = str(result_root / "eval_results.jsonl")
config["media"]["collision_dir"] = str(result_root / "collisions")
Path(os.environ["RUNTIME_CONFIG"]).write_text(
    json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
)
PY

server_pid=""
xvfb_pid=""
cleanup() {
  if [[ -n "${server_pid}" ]]; then kill "${server_pid}" >/dev/null 2>&1 || true; fi
  if [[ -n "${xvfb_pid}" ]]; then kill "${xvfb_pid}" >/dev/null 2>&1 || true; fi
}
trap cleanup EXIT INT TERM

export DISPLAY="${DISPLAY:-:9}"
if command -v Xvfb >/dev/null 2>&1; then
  Xvfb "${DISPLAY}" -screen 0 1024x768x24 >"${RESULT_ROOT}/xvfb.log" 2>&1 &
  xvfb_pid=$!
fi

echo "[openuav-worker] config=${CONFIG} env=${EVAL_ENV:-from-config} samples=${MAX_SAMPLES} result=${RESULT_ROOT}"
nvidia-smi || true

CUDA_VISIBLE_DEVICES="${MODEL_GPU}" \
PYTHONPATH="${AEROLOOP_ROOT}/src" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONNOUSERSITE=1 \
"${SERVER_PYTHON}" -m aeroloop.server_cli openuav \
  --repo-root "${AB_EX_ROOT}/OpenUAV" \
  --ckpt-dir "${AB_EX_ROOT}/OpenUAV/ckpt" \
  --traj-model-path "${AB_EX_ROOT}/OpenUAV/ckpt/model_2.pth" \
  --device cuda:0 \
  --stop-norm-threshold "${OPENUAV_STOP_THRESHOLD:-0.5}" \
  --history-size "${OPENUAV_HISTORY_SIZE:-4}" \
  --host 127.0.0.1 --port "${SERVER_PORT}" \
  >"${RESULT_ROOT}/openuav_server.log" 2>&1 &
server_pid=$!

export SERVER_PORT
"${EVAL_PYTHON}" - <<'PY'
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
            print(f"[openuav-worker] health: {payload}", flush=True)
            break
    except Exception as exc:
        print(f"[openuav-worker] waiting for model: {exc}", flush=True)
    time.sleep(10)
else:
    raise TimeoutError("OpenUAV did not become healthy in 900 seconds")
PY

cd "${RESULT_ROOT}"
CUDA_VISIBLE_DEVICES="${SIM_GPU}" \
PYTHONPATH="${AEROLOOP_ROOT}/src" \
"${EVAL_PYTHON}" -m aeroloop run \
  --config "${RUNTIME_CONFIG}" \
  --output-jsonl "${RESULT_ROOT}/eval_results.jsonl" \
  --headless --no-video

"${EVAL_PYTHON}" "${AEROLOOP_ROOT}/scripts/validate_eval_jsonl.py" \
  "${RESULT_ROOT}/eval_results.jsonl" --expected "${MAX_SAMPLES}"
