# Model HTTP servers

Install AeroLoop into every model environment first:

```bash
cd /path/to/AeroLoop
python -m pip install -e . --no-deps
```

Every server exposes:

- `GET /health`
- `POST /v1/reset`
- `POST /v1/predict`

The HTTP contract, error handling, and WorldVLN cache/unit conversion are
covered by automated tests. GPU loading must still be smoke-tested inside each
model's own environment; those dependencies are deliberately not installed in
the lightweight AeroLoop environment.

The launch scripts accept `PYTHON_BIN`, path, host, port, device, and model
settings through environment variables.

## AerialVLA

```bash
PYTHON_BIN=/path/to/aerialvla/bin/python bash scripts/serve_aerialvla.sh
curl http://127.0.0.1:18101/health
```

The adapter uses the training codec (`NUM_BINS=99`). Native
`[forward,down,d_yaw,LAND]` is returned canonically as
`[forward,0,down,d_yaw,stop]`; lateral motion cannot be recovered because it
was not represented by the model target.

## OpenUAV

```bash
PYTHON_BIN=/path/to/llamauav/bin/python bash scripts/serve_openuav.sh
curl http://127.0.0.1:18102/health
```

The adapter follows the complete OpenUAV path: the LLM predicts native
`[unit_xyz,norm]`, then the trajectory-completion checkpoint refines the target
into seven cumulative body-frame points. The server differences those points
into canonical action deltas. A near-zero LLM norm produces one explicit stop
action and bypasses the refiner.

The loader expands the 32,000-token Vicuna base with `<wp>` and `<his>`, removes
PEFT wrapper prefixes, and validates both navigation-head and trajectory-head
weights. `OPENUAV_CKPT` is the directory containing `final/`, `model_zoo/`, and
normally `model_2.pth`; use `OPENUAV_TRAJ_CKPT` to override the latter.

## DualVLN

```bash
PYTHON_BIN=/path/to/internnav/bin/python bash scripts/serve_dualvln.sh
curl http://127.0.0.1:18103/health
```

The adapter calls `generate_latents` and `generate_traj`. Stage-2 output is a
cumulative 32-point body-frame trajectory, so the server differences it into
32 per-step translations before returning the canonical chunk. The bundled
stage-2 directory has no `preprocessor_config.json`; the adapter intentionally
uses the matching stage-1 Qwen image processor with the stage-2 tokenizer.

## OmniNav

```bash
PYTHON_BIN=/path/to/compatible/python bash scripts/serve_omninav.sh
curl http://127.0.0.1:18105/health
```

The adapter targets the OpenFly-trained 3-D OmniNav checkpoint. It reproduces
the training input exactly: five uniformly sampled historical front-camera
frames and body-frame positions, followed by the current front view. The model
returns five cumulative body-frame XYZ waypoints and five arrive logits. The
server differences the waypoints into canonical action deltas, derives yaw
from XY translation, and applies sigmoid to each arrive logit.

OmniNav must import the modified Transformers source bundled under
`train_code/transformers-main`; use an environment with Torch, Accelerate,
Pillow, and `tokenizers>=0.21,<0.22`. On the current workspace,
`internnav-infer` is compatible. The default checkpoint is
`OmniNav/checkpoint-19805`, and both paths can be overridden with
`OMNINAV_ROOT` and `OMNINAV_CKPT`.

## WorldVLN

First start the native WorldVLN service in its environment:

```bash
cd /path/to/WorldVLN
INFINITY_CKPT=/path/to/backbone_stage1.pth \
STAGE2_LATENT2ACTION_CKPT=/path/to/action_decoder.pt \
VAE_PATH=/path/to/vae/model.safetensors \
STAGE2_VAE_PATH=/path/to/vae/model.safetensors \
T5_PATH=/path/to/flan-t5-xl-directory \
bash infer/run_server.sh
```

Then start the canonical proxy:

```bash
WORLDVLN_UPSTREAM_URL=http://127.0.0.1:8001 \
PYTHON_BIN=/path/to/worldvln/bin/python \
bash scripts/serve_worldvln_proxy.sh
curl http://127.0.0.1:18104/health
```

WorldVLN predicts a complete native segment. The proxy returns one cached
action per AeroLoop call while collecting real frames, then asks the upstream
server for the next segment. `metadata.native_replan` distinguishes a real
WorldVLN replan from a cached action. Upstream cm/degree actions are converted
to metres/radians.

## Evaluation configuration

The files under `configs/models/` inherit the generic HTTP template. Combine
one of them with a built-in simulator config or replace the simulator and
episode source with extensions from your integration package. AeroLoop ships
generic AirSim, GS-AirSim, and UnrealCV adapters but does not distribute scene
executables.

The primary benchmark may retain `execution_horizon: 1`; the WorldVLN proxy
will still disclose its intrinsic segment-level open-loop behavior in metadata.

DualVLN, OpenUAV, and OmniNav predict local XYZ trajectories without a separate yaw
channel. Their adapters derive `d_yaw=atan2(dy,dx)` in the canonical body
frame. AerialVLA and WorldVLN retain their native explicit yaw predictions.

```bash
aeroloop run --config configs/models/aerialvla.yaml
```

The command becomes runnable after the two integration entry points in
`configs/http.yaml` are replaced.
