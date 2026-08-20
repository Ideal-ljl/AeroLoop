#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AEROLOOP_ROOT="${AEROLOOP_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
PI0_ROOT="${PI0_ROOT:-/mnt/petrelfs/youzhongrui/v2/lerobot-openfly-pi0}"
PI0_CHECKPOINT="${PI0_CHECKPOINT:-/mnt/petrelfs/youzhongrui/v2/Ab_ex/checkpoints/pi0-openfly-16g-bs64-v2-0723/030000/pretrained_model}"
PI0_TOKENIZER="${PI0_TOKENIZER:-${PI0_ROOT}/assets/paligemma_tokenizer}"
EVAL_PYTHON="${EVAL_PYTHON:-/mnt/petrelfs/youzhongrui/miniconda3/envs/qwen/bin/python}"
SERVER_PYTHON="${SERVER_PYTHON:-${PI0_ROOT}/.venv/bin/python}"
CONFIG="${CONFIG:-${AEROLOOP_ROOT}/configs/jobs/pi0_openfly.yaml}"
SOURCE_EVAL_CONFIG="${SOURCE_EVAL_CONFIG:-${AEROLOOP_ROOT}/configs/datasets/openfly_fixed_11x100_recovered.json}"
RESULT_ROOT="${RESULT_ROOT:?RESULT_ROOT is required}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
SERVER_PORT="${SERVER_PORT:-18106}"
MODEL_GPU="${MODEL_GPU:-0}"
SIM_GPU="${SIM_GPU:-0}"
EVAL_ENV="${EVAL_ENV:-env_airsim_18}"
PI0_SEED="${PI0_SEED:-2026}"
PI0_INFERENCE_STEPS="${PI0_INFERENCE_STEPS:-10}"
PI0_HISTORY_STEPS="${PI0_HISTORY_STEPS:-1}"

mkdir -p "${RESULT_ROOT}"
exec 9>"${RESULT_ROOT}/worker.lock"
flock 9

PROGRESS_JSONL="${RESULT_ROOT}/progress.jsonl"
FINAL_JSONL="${RESULT_ROOT}/eval_results.jsonl"
RUNTIME_CONFIG="${RESULT_ROOT}/runtime_config.json"
REMAINING_DATASET="${RESULT_ROOT}/remaining_dataset.json"
STATE_JSON="${RESULT_ROOT}/resume_state.json"
export AEROLOOP_ROOT CONFIG SOURCE_EVAL_CONFIG RESULT_ROOT MAX_SAMPLES PROGRESS_JSONL FINAL_JSONL EVAL_ENV
export RUNTIME_CONFIG REMAINING_DATASET STATE_JSON SERVER_PORT
export PI0_HISTORY_STEPS

if [[ ! -x "${SERVER_PYTHON}" ]]; then
  echo "PI0 server Python is missing: ${SERVER_PYTHON}" >&2
  exit 2
fi
if [[ ! -f "${PI0_CHECKPOINT}/model.safetensors" ]]; then
  echo "PI0 checkpoint is missing: ${PI0_CHECKPOINT}" >&2
  exit 2
fi

compact_results() {
  PYTHONPATH="${AEROLOOP_ROOT}/src" "${EVAL_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

progress = Path(os.environ["PROGRESS_JSONL"])
final = Path(os.environ["FINAL_JSONL"])
episodes = {}
if progress.exists():
    with progress.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("type") != "episode" or not row.get("episode_id"):
                continue
            previous = episodes.get(row["episode_id"])
            valid = not row.get("error") and row.get("termination_reason") != "error"
            previous_valid = previous and not previous.get("error") and previous.get("termination_reason") != "error"
            if valid or not previous_valid:
                episodes[row["episode_id"]] = row
tmp = final.with_suffix(".tmp")
with tmp.open("w", encoding="utf-8") as handle:
    for episode_id in sorted(episodes):
        handle.write(json.dumps(episodes[episode_id], ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
tmp.replace(final)
print(f"[pi0-worker] compacted {len(episodes)} unique episodes to {final}", flush=True)
PY
}

PYTHONPATH="${AEROLOOP_ROOT}/src" "${EVAL_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

from aeroloop.config import load_config

config = load_config(os.environ["CONFIG"])
result_root = Path(os.environ["RESULT_ROOT"])
progress = Path(os.environ["PROGRESS_JSONL"])
max_samples = int(os.environ["MAX_SAMPLES"])
eval_env = os.environ["EVAL_ENV"]

completed = set()
if progress.exists():
    with progress.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                row.get("type") == "episode"
                and row.get("episode_id")
                and not row.get("error")
                and row.get("termination_reason") != "error"
            ):
                completed.add(row["episode_id"])

benchmark = config["benchmark"]
kwargs = benchmark["kwargs"]
source_path = Path(os.environ["SOURCE_EVAL_CONFIG"])
kwargs["path"] = str(source_path)
kwargs["include_envs"] = eval_env
source_rows = json.loads(source_path.read_text(encoding="utf-8"))

selected = []
for row in source_rows:
    path = Path(row["path"])
    env_name = next((part for part in path.parts if part.startswith("env_")), path.parent.name)
    if env_name != eval_env:
        continue
    episode_id = f"openfly:{env_name}:{path.name}"
    selected.append((episode_id, row))
    if len(selected) >= max_samples:
        break

remaining = [row for episode_id, row in selected if episode_id not in completed]
if len(selected) != max_samples:
    raise ValueError(f"{eval_env} selected {len(selected)} samples from {source_path}, expected {max_samples}")
Path(os.environ["REMAINING_DATASET"]).write_text(
    json.dumps(remaining, ensure_ascii=False, indent=2), encoding="utf-8"
)
kwargs["path"] = os.environ["REMAINING_DATASET"]
kwargs["limit"] = len(remaining)
common_camera = {"name": "front", "width": 896, "height": 896, "fov": 90}
runtime_root = Path(os.environ["AEROLOOP_ROOT"]) / ".runtime" / "envs"
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
    # GS render coordinates are smaller than the OpenFly dataset coordinates.
    # Match the legacy OpenFly canonical-position / point-cloud scale conversion while
    # keeping model state and benchmark metrics in the original dataset frame.
    gs_position_scales = {
        "env_gs_ecust": 1.0 / 5.6,
        "env_gs_nwpu01": 1.0 / 6.65,
        "env_gs_nwpu02": 1.0 / 5.15,
        "env_gs_sjtu01": 1.0 / 5.42,
        "env_gs_sjtu02": 1.0 / 4.75,
    }
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
            "position_scale": gs_position_scales[eval_env],
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
            "startup_grace_s": 20,
            "cameras": [common_camera],
            "camera_ids": {"front": 1},
            "collision_query": None,
        },
    }
else:
    raise ValueError(f"unsupported OpenFly evaluation environment: {eval_env!r}")

config["policy"]["kwargs"].update(
    {
        "url": f"http://127.0.0.1:{os.environ['SERVER_PORT']}/v1/predict",
        "reset_url": f"http://127.0.0.1:{os.environ['SERVER_PORT']}/v1/reset",
        "views": ["front"],
        "observation_fields": ["images", "state"],
        "state_source": "relative",
        "history_steps": int(os.environ["PI0_HISTORY_STEPS"]),
    }
)
config["rollout"]["execution_horizon"] = 10
config["simulator"]["kwargs"]["log_path"] = str(result_root / "simulator.log")
config["output"]["jsonl"] = os.environ["PROGRESS_JSONL"]
config["media"]["collision_dir"] = str(result_root / "collisions")
Path(os.environ["RUNTIME_CONFIG"]).write_text(
    json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
)
Path(os.environ["STATE_JSON"]).write_text(
    json.dumps(
        {
            "requested": len(selected),
            "completed": len(completed.intersection(episode_id for episode_id, _ in selected)),
            "remaining": len(remaining),
        },
        indent=2,
    ),
    encoding="utf-8",
)
print(f"[pi0-worker] requested={len(selected)} completed={len(completed)} remaining={len(remaining)}", flush=True)
PY

remaining="$("${EVAL_PYTHON}" -c "import json; print(json.load(open('${STATE_JSON}'))['remaining'])")"
if [[ "${remaining}" == "0" ]]; then
  compact_results
  "${EVAL_PYTHON}" "${AEROLOOP_ROOT}/scripts/validate_eval_jsonl.py" \
    "${FINAL_JSONL}" --expected "${MAX_SAMPLES}"
  exit 0
fi

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

echo "[pi0-worker] config=${CONFIG} env=${EVAL_ENV} samples=${MAX_SAMPLES} result=${RESULT_ROOT}"
"${SERVER_PYTHON}" -c \
  "import sys, torch, transformers; assert sys.version_info[:2] == (3, 12); assert transformers.__version__ == '5.5.4'; print('[pi0-worker] python/torch/transformers', sys.version.split()[0], torch.__version__, transformers.__version__)"
nvidia-smi || true

CUDA_VISIBLE_DEVICES="${MODEL_GPU}" \
PYTHONPATH="${AEROLOOP_ROOT}/src:${PI0_ROOT}/src" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONNOUSERSITE=1 \
"${SERVER_PYTHON}" -m aeroloop.server_cli pi0 \
  --repo-root "${PI0_ROOT}" \
  --ckpt-dir "${PI0_CHECKPOINT}" \
  --tokenizer-dir "${PI0_TOKENIZER}" \
  --device cuda \
  --dtype bfloat16 \
  --seed "${PI0_SEED}" \
  --inference-steps "${PI0_INFERENCE_STEPS}" \
  --host 127.0.0.1 --port "${SERVER_PORT}" \
  >"${RESULT_ROOT}/pi0_server.log" 2>&1 &
server_pid=$!

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
        if (
            payload.get("status") == "ok"
            and payload.get("checkpoint_verified") is True
            and payload.get("observation_steps") == int(os.environ["PI0_HISTORY_STEPS"])
        ):
            print(f"[pi0-worker] health: {payload}", flush=True)
            break
    except Exception as exc:
        print(f"[pi0-worker] waiting for model: {exc}", flush=True)
    time.sleep(10)
else:
    raise TimeoutError("PI0 did not become healthy and verified in 900 seconds")
PY

cd "${RESULT_ROOT}"
CUDA_VISIBLE_DEVICES="${SIM_GPU}" \
PYTHONPATH="${AEROLOOP_ROOT}/src" \
"${EVAL_PYTHON}" -m aeroloop run \
  --config "${RUNTIME_CONFIG}" \
  --output-jsonl "${PROGRESS_JSONL}" \
  --headless --no-video

compact_results
"${EVAL_PYTHON}" "${AEROLOOP_ROOT}/scripts/validate_eval_jsonl.py" \
  "${FINAL_JSONL}" --expected "${MAX_SAMPLES}"
