#!/usr/bin/env bash
set -euo pipefail

AB_EX_ROOT="${AB_EX_ROOT:-/home/liujunli/ceph/liujunli/Ab_ex}"
CONDA_ROOT="${CONDA_ROOT:-/mnt/petrelfs/youzhongrui/miniconda3}"

echo "[inventory] checkpoint files"
find "${AB_EX_ROOT}/ckpt" -maxdepth 5 -type f \
  \( -name 'adapter_model.safetensors' -o -name 'adapter_model.bin' \
     -o -name 'non_lora_trainables.bin' -o -name 'mm_projector.bin' \
     -o -name 'model.safetensors.index.json' -o -name 'backbone_stage1.pth' \
     -o -name 'action_decoder.pt' \) \
  -printf '%p %s\n' | sort

echo "[inventory] OpenUAV exports outside ckpt"
find "${AB_EX_ROOT}/OpenUAV" -maxdepth 8 -type f \
  \( -name 'adapter_model.safetensors' -o -name 'adapter_model.bin' \
     -o -name 'non_lora_trainables.bin' -o -name 'mm_projector.bin' \) \
  -printf '%p %s\n' | sort

echo "[inventory] Python environments"
for env_name in base qwen uavflow internnav-infer aerialvla_openfly llamauav internnav worldvln; do
  if [[ "${env_name}" == base ]]; then
    python_bin="${CONDA_ROOT}/bin/python"
  else
    python_bin="${CONDA_ROOT}/envs/${env_name}/bin/python"
  fi
  if [[ -x "${python_bin}" ]]; then
    printf '%s ' "${env_name}"
    "${python_bin}" -c 'import sys; print(sys.version.split()[0])'
  else
    echo "${env_name} MISSING"
  fi
done

echo "[inventory] model runtime imports"
for env_name in qwen uavflow internnav-infer; do
  python_bin="${CONDA_ROOT}/envs/${env_name}/bin/python"
  printf '%s ' "${env_name}"
  "${python_bin}" - <<'PY'
modules = ["torch", "transformers", "peft", "fastapi", "uvicorn", "open3d", "airsim"]
for module in modules:
    try:
        __import__(module)
        print(f"{module}=ok", end=" ")
    except Exception:
        print(f"{module}=missing", end=" ")
print()
PY
done
