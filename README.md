# UAVEval

UAVEval is a model-agnostic closed-loop evaluation platform for UAV
vision-language navigation. It separates simulator rendering, model inference,
kinematic action execution, metrics, and recording so that policies with
different runtimes and native action spaces can be compared under one protocol.

## Core protocol

The canonical observation contains:

- current RGB `uint8` image;
- absolute world pose `[x, y, z, yaw]`;
- start-frame-relative state `[x, y, z, yaw]`;
- instruction and episode metadata;
- optional image/action/state history managed by the rollout engine.

The canonical action is:

```text
[dx_body, dy_body, dz_body, d_yaw, stop_probability]
```

Translation is expressed in the drone body frame. Strafing does not change
yaw; heading changes only through `d_yaw`. Four-column OpenFly actions
`[dx,dy,dz,stop]` remain accepted and imply `d_yaw=0`.

## Architecture

```text
Episode source ─┐
                v
          RolloutRunner <──── PolicyAdapter (local / HTTP / plugin)
                │
                v
       EnvironmentAdapter (AirSim / GS-AirSim / UE / mock)
                │
                ├── MetricSuite: SR, SPL, OSR, collision, stop, distances
                └── JSONL recorder: run config, episode traces, summaries
```

Model and simulator dependencies are deliberately not imported by the core
package. A model can run in its own conda environment behind the canonical HTTP
interface.

## Quick smoke test

No NumPy, PyTorch, simulator, or GPU is required for the mock run:

```bash
cd /mnt/petrelfs/youzhongrui/v2/UAVEval
PYTHONPATH=src python -m unittest discover -s tests -v
rm -rf eval_results
PYTHONPATH=src python -m uav_eval run --config configs/mock.yaml
```

Install as a package when the environment has a working pip:

```bash
pip install -e .
uav-eval run --config configs/mock.yaml
```

## AirBrain evaluation

The AirBrain adapter reuses `scripts/env_bridge.py`, including the AirSim,
GS-AirSim, UnrealCV, point-cloud surface-distance, and collision components.
UAVEval owns pose integration and always feeds models state in dataset units;
the GS `pcd_scale_ratio` is applied only when setting renderer pose.

Edit `configs/airbrain_http.yaml`, start a model service, then run:

```bash
PYTHONPATH=src python -m uav_eval run --config configs/airbrain_http.yaml
```

The AirBrain runtime additionally needs its normal simulator dependencies such
as NumPy, OpenCV, Open3D, AirSim/UnrealCV, and the environment assets.

## HTTP policy contract

`POST /v1/reset` receives:

```json
{
  "episode_id": "env:episode",
  "env_name": "env_airsim_18",
  "instruction": "fly toward the gray tower"
}
```

`POST /v1/predict` receives:

```json
{
  "episode_id": "env:episode",
  "env_name": "env_airsim_18",
  "instruction": "fly toward the gray tower",
  "step": 0,
  "state": [0.0, 0.0, 0.0, 0.0],
  "pose": [12.0, 8.0, 5.0, 1.57],
  "image_base64": "<PNG>",
  "image_format": "png_rgb"
}
```

The response must contain a non-empty action chunk:

```json
{
  "actions": [
    [0.5, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.0, 0.0, 0.0]
  ],
  "metadata": {"model": "example", "native_chunk_size": 2}
}
```

Four-column rows `[dx,dy,dz,stop]` are also accepted. Each model service is
responsible for converting native outputs, units, coordinate conventions, and
stop semantics into the canonical action.

A dependency-free reference server is included:

```bash
python examples/mock_http_server.py --port 18080
```

## Fair chunk execution

`rollout.execution_horizon` controls how much of each predicted chunk is used:

- `1`: observe and replan every action; recommended for the primary comparison;
- positive `K`: execute at most K actions before replanning;
- `null`: consume the model's complete native chunk open-loop.

Discarded actions are not carried into the next inference call.

## Metrics

Every episode records endpoint 2D/3D distance, building-center 2D distance,
point-cloud surface distance, collision, stop behavior, inference latency, path
length, SR, SPL, and OSR. The main success distance can be configured as:

- `endpoint_2d`
- `endpoint_3d`
- `surface`
- `legacy_min` (compatibility with `test_full_his.py`)

Stop is reported independently as `stop_called`, `stop_success`, and
`premature_stop`. `require_stop_for_success` controls whether SR requires an
explicit stop.

## Policy adapters

- `mock`: deterministic dependency-free integration test.
- `http`: recommended boundary for real models in different conda envs.
- `plugin`: loads `python.module:ClassName` implementing `reset()` and
  `predict()`.
- `ckpt_harness`: compatibility with `Ab_ex/ckpt/harness`; concrete wrappers
  must be repaired and validated before benchmark use.

The next integration layer should provide one HTTP server per AerialVLA,
DualVLN, OpenUAV, and WorldVLN environment. The benchmark runner and simulator
do not change when a new model is added.
