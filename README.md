# UAVEval

UAVEval is an extensible closed-loop evaluation platform for UAV
vision-language navigation. It gives simulators and models one small protocol,
while keeping their heavyweight dependencies in separate environments.

## What it standardizes

- observation: instruction, named RGB views, world pose, start-relative state,
  and optional model-specific state;
- action: `[dx_body, dy_body, dz_body, d_yaw, stop_probability]`;
- rollout: collision/stop handling and fair action-chunk execution;
- evaluation: SR, SPL, OSR, distances, latency, JSONL traces, and optional video.

Environments, policies, episode sources, metrics, observers, and camera layouts
are replaceable. Existing four-column actions `[dx,dy,dz,stop]` and the legacy
single `observation.rgb` view remain supported.

## Install and run

```bash
pip install -e .
uav-eval run --config configs/mock.yaml
```

The mock run needs no simulator, GPU, NumPy, or PyTorch. Optional features:

```bash
pip install -e '.[media]'     # OpenCV visualization and MP4 output
pip install -e '.[airbrain]'  # AirBrain runtime helpers
uav-eval doctor
```

## Configuration

```yaml
benchmark:
  source: inline
  episodes:
    - episode_id: demo
      env_name: mock
      instruction: Fly to the target.
      target_position: [10, 0, 0]

environment:
  type: mock
  kwargs:
    cameras: standard  # front, back, left, right, down

policy:
  type: mock
  kwargs:
    action: [1, 0, 0, 0, 0]

rollout:
  max_steps: 200
  execution_horizon: 1  # null consumes each model's native action chunk

metrics:
  success_distance: 25
  distance_mode: endpoint_3d
```

Use `cameras: standard`, a list of standard names, or custom body-relative
camera definitions:

```yaml
cameras:
  - front
  - name: gimbal
    position: [0.2, 0, -0.1]
    yaw_degrees: 30
    pitch_degrees: -25
    fov: 90
```

An HTTP model selects only the views it needs; the environment must render
those names:

```yaml
policy:
  type: http
  kwargs:
    url: http://127.0.0.1:18080/v1/predict
    reset_url: http://127.0.0.1:18080/v1/reset
    views: [front, down]
```

## Architecture

```text
Episode source ──> RolloutRunner <── Policy (local / HTTP / plugin)
                         │
                         v
                 Environment (simulator / plugin)
                         │
                  metrics + observers
                         │
                  JSONL + media artifacts
```

Real models should normally run behind the HTTP boundary in their own conda
environment. UAVEval includes adapters for AerialVLA, OpenUAV, DualVLN, and
WorldVLN; see [model server setup](docs/MODEL_SERVERS.md).

## Extend UAVEval

A component can be referenced directly as `package.module:Object`, or published
through these Python entry-point groups:

- `uav_eval.environments`
- `uav_eval.policies`
- `uav_eval.episode_sources`
- `uav_eval.metrics`

The stable interfaces are `EnvironmentAdapter`, `PolicyAdapter`, `Metric`, and
`RolloutObserver`. See [extension guide](docs/EXTENDING.md) for minimal working
implementations and [HTTP API](docs/HTTP_API.md) for multi-view requests.

## Development

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
ruff check src tests
```

Output rows are documented in [the JSONL schema](docs/OUTPUT_SCHEMA.md). Before
benchmarking a new model, complete the [adapter checklist](docs/MODEL_ADAPTER_CHECKLIST.md).
Contributions are described in [CONTRIBUTING.md](CONTRIBUTING.md); the project
is licensed under [Apache-2.0](LICENSE).
