# Model HTTP servers

Install UAVEval into every model environment first:

```bash
cd /mnt/petrelfs/youzhongrui/v2/UAVEval
python -m pip install -e . --no-deps
```

Every server exposes:

- `GET /health`
- `POST /v1/reset`
- `POST /v1/predict`

The HTTP contract, error handling, and WorldVLN cache/unit conversion are
covered by automated tests. GPU loading must still be smoke-tested inside each
model's own environment; those dependencies are deliberately not installed in
the lightweight UAVEval environment.

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

The adapter follows the original waypoint-forward path, but loads the 32,000
token Vicuna base before expanding it with `<wp>` and `<his>`. It also removes
PEFT wrapper prefixes from exported weights. The server validates the waypoint,
history, and added-token weights before allocating the base model and refuses
to start with randomly initialized navigation heads.

The currently bundled `ckpt/OpenUAV/final/mm_projector.bin` contains only the
vision projector and Q-Former. It does **not** contain the trained waypoint,
history, or added-token weights, and has no LoRA adapter, so it is intentionally
rejected. Point `OPENUAV_CKPT` at a complete export before starting this server.

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

## WorldVLN

First start the native WorldVLN service in its environment:

```bash
cd /path/to/WorldVLN
INFINITY_CKPT=/path/to/backbone_stage1.pth \
STAGE2_LATENT2ACTION_CKPT=/path/to/action_decoder.pt \
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
action per UAVEval call while collecting real frames, then asks the upstream
server for the next segment. `metadata.native_replan` distinguishes a real
WorldVLN replan from a cached action. Upstream cm/degree actions are converted
to metres/radians.

## Benchmark configuration

The files under `configs/models/` inherit the common AirBrain configuration and
can be run directly after editing dataset paths. The primary benchmark may retain
`execution_horizon: 1`; the WorldVLN proxy will still disclose its intrinsic
segment-level open-loop behavior in metadata.

DualVLN and OpenUAV predict local XYZ trajectories without a separate yaw
channel. Their adapters derive `d_yaw=atan2(dy,dx)`, matching AirBrain's
trajectory-heading update. AerialVLA and WorldVLN retain their native explicit
yaw predictions.

```bash
uav-eval run --config configs/models/aerialvla.yaml
```
