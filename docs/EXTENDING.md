# Extending AeroLoop

Extensions may be referenced by `module:Object` in YAML while developing. A
reusable package should publish a Python entry point so users can select it by
name without modifying AeroLoop.

## Simulator

A simulator adapter owns rendering, physics, process lifecycle, sensors, and
coordinate conversion. AeroLoop includes direct `airsim`, `gs_airsim`, and
`unrealcv` adapters. Use an extension for another SDK or for scene-specific
capabilities not covered by those generic adapters.

```python
from aeroloop import Observation, SimulatorAdapter
from aeroloop.types import Transition

class MySimulator(SimulatorAdapter):
    def __init__(self, env_name, cameras="front", **kwargs):
        self.env_name = env_name

    def reset(self, episode):
        ...
        return Observation(
            rgb=front_image,
            images={"front": front_image, "down": down_image},
            camera_specs={camera.name: camera for camera in self.camera_specs},
            primary_view="front",
            pose=pose,
            relative_state=(0, 0, 0, 0),
            step_index=0,
            auxiliary_state={"velocity": [0, 0, 0]},
        )

    def execute(self, action):
        ...
        return Transition(observation, collision=False, info={})
```

```yaml
simulator:
  type: custom
  entrypoint: my_package.simulator:MySimulator
  kwargs: {cameras: standard}
```

Do not silently change requested camera resolution or coordinate conventions.
Document supported views and units, and return all images as RGB `uint8`.

## Policy

Subclass `PolicyAdapter` and implement `reset()` and `predict()`. `PolicyInput`
contains the current observation plus primary-image, all-view, action, and state
history. Return an `ActionChunk` in canonical body-frame units. For models with
isolated dependencies, prefer the HTTP API.

## Metric

Custom metrics receive every executed transition and are namespaced in output.

```python
from aeroloop import Metric

class EnergyMetric(Metric):
    name = "energy"

    def reset(self, episode):
        self.total = 0.0

    def update(self, before, after, *, action, collision, info):
        self.total += action.dx**2 + action.dy**2 + action.dz**2

    def finalize(self, termination, final_pose, step_count):
        return {"total": self.total}
```

```yaml
metrics:
  success_distance: 25
  distance_mode: endpoint_3d
  custom:
    - entrypoint: my_package.metrics:EnergyMetric
      kwargs: {}
```

The result key is `energy/total`. Metric names must be unique. Exceptions fail
the episode visibly instead of producing partial or misleading metrics.

## Episode source

An episode-source extension is a callable returning `EpisodeSpec` objects:

```yaml
benchmark:
  source: custom
  entrypoint: my_package.dataset:load_episodes
  kwargs: {split: test}
```

## Data collector / observer

Observers receive the initial observation, every executed step, and the final
result. They are the preferred place to save simulator-specific sensors,
labels, telemetry, or dataset manifests without adding those dependencies to
AeroLoop.

```python
from aeroloop import RolloutObserver

class DatasetCollector(RolloutObserver):
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def on_step(self, episode, observation, record):
        save_sample(self.output_dir, episode, observation, record)
        return True
```

```yaml
observers:
  - entrypoint: my_package.collectors:DatasetCollector
    kwargs: {output_dir: collected_data}
```

## Package entry points

```toml
[project.entry-points."aeroloop.simulators"]
my-simulator = "my_package.simulator:MySimulator"

[project.entry-points."aeroloop.metrics"]
energy = "my_package.metrics:EnergyMetric"

[project.entry-points."aeroloop.observers"]
dataset = "my_package.collectors:DatasetCollector"
```

Then use `type: my-simulator` or `type: energy`. Policy and episode-source
groups follow the same pattern. The old `EnvironmentAdapter`, `environment:`
config key, and `aeroloop.environments` group remain compatibility aliases.
Test extensions against the public protocols and include a deterministic fixture.
