# Simulator integration contract

AeroLoop is the interaction layer. Its optional built-in SDK adapters directly
control AirSim-compatible and UnrealCV scenes; other integrations implement
`SimulatorAdapter` and may publish it through the `aeroloop.simulators`
entry-point group. Scene executables remain external assets in both cases.

## Lifecycle

1. `reset(episode)` starts or resets the scene and returns the first observation.
2. `execute(action)` applies one canonical action and returns a transition.
3. `close()` releases simulator processes, sockets, and rendering resources.

`reset` and `execute` must be deterministic under the integration's documented
seed settings. Simulator failures should raise exceptions; they must not be
converted into stop actions or successful episodes.

## Observation

An `Observation` contains:

- `images`: RGB `uint8` arrays keyed by view name;
- `camera_specs`: body-relative extrinsics and available intrinsics;
- `pose`: absolute world pose `[x,y,z,yaw]`;
- `relative_state`: start-frame-relative `[x,y,z,yaw]`;
- `auxiliary_state`: integration-defined JSON-compatible sensor state;
- `info`: simulator metadata used locally by metrics and observers.

The standard view names are `front`, `back`, `left`, `right`, and `down`.
Custom view names and camera poses are allowed. An integration must reject
unsupported resolution/FOV requests instead of silently changing them.

The integration should accept requested cameras and sensors through its
constructor configuration and avoid rendering disabled streams. HTTP payload
filtering is a second boundary and does not compensate for expensive unused
rendering.

## Action

The canonical action is
`[dx_body, dy_body, dz_body, d_yaw, stop_probability]`. Translation uses the
UAV body frame and yaw uses radians. A simulator integration is responsible for
converting these values to its native controls and units.

`Transition.info` may expose values such as `surface_distance`; custom metrics
should document any required keys rather than making them core requirements.
Collision detection remains the simulator integration's responsibility and is
reported as `Transition.collision`. AeroLoop aggregates collision rate, can end
the rollout immediately, and can save the first collision frame.

## Data collection

AeroLoop records canonical observations, requested actions, resulting poses,
timings, collisions, metrics, and observer artifacts. Simulator-specific raw
sensors or labels should be emitted by a custom `RolloutObserver` so the core
does not acquire simulator dependencies. Observer extensions can be loaded from
YAML through the `aeroloop.observers` entry-point group.
