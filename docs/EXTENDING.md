# Extending UAVEval

Extensions may be referenced by `module:Object` in YAML while developing. A
reusable package should publish a Python entry point so users can select it by
name without modifying UAVEval.

## Environment

An environment owns rendering and simulator interaction. UAVEval owns the
canonical action and rollout protocol.

```python
from uav_eval import EnvironmentAdapter, Observation
from uav_eval.types import Transition

class MyEnvironment(EnvironmentAdapter):
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
environment:
  type: custom
  entrypoint: my_package.environment:MyEnvironment
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
from uav_eval import Metric

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

## Package entry points

```toml
[project.entry-points."uav_eval.environments"]
my-simulator = "my_package.environment:MyEnvironment"

[project.entry-points."uav_eval.metrics"]
energy = "my_package.metrics:EnergyMetric"
```

Then use `type: my-simulator` or `type: energy`. Policy and episode-source
groups follow the same pattern. Test extensions against the public protocols
and include at least one deterministic fixture.
