# Direct simulator control

AeroLoop contains optional SDK adapters for AirSim-compatible and UnrealCV
scenes. These adapters replace project-specific bridge scripts: they start a
packaged scene, connect to its SDK, convert canonical coordinates, capture
cameras, execute actions, report collision state, and stop the process.

Scene executables are external assets and are never copied into the package.
For a source checkout, the recommended local location is `.runtime/envs/`,
which is ignored by Git and can be referenced with relative `env_root` paths.

## AirSim

```yaml
simulator:
  type: airsim
  kwargs:
    env_root: /data/simulators/airsim
    launch_script: LinuxNoEditor/start.sh
    port: 41451
    cameras: [front]
    camera_names: {front: front_custom}
    configure_camera_poses: false
    position_sign: [1, -1, -1]
    yaw_sign: -1
    ignore_collision: false
```

For `env_name: env_airsim_16`, this resolves the executable below
`/data/simulators/airsim/env_airsim_16`. `settings.json` is located
automatically and its `ApiServerPort` is updated before launch.

Set `launch: false` to connect to a scene process managed by another scheduler.
`ignore_collision: false` asks AirSim to honor collision during pose updates,
and `simGetCollisionInfo` supplies `Transition.collision`.
`configure_camera_poses: false` preserves the extrinsics packaged in
`settings.json`; enable it only when the YAML `CameraSpec` should replace them.

## GS AirSim

`type: gs_airsim` uses the same implementation with the GS defaults:

- SDK module `airsim_ue5`;
- `VehicleClient`;
- `gs.sh` launcher;
- `[x, y, -z]` position transform and positive yaw;
- horizontal image flip compatible with the existing GS datasets.

Every default can be overridden in YAML. The `airsim_ue5` wheel used by a
scene can be installed in the simulator/AeroLoop runtime without affecting the
model's HTTP environment.

## UnrealCV

```yaml
simulator:
  type: unrealcv
  kwargs:
    env_root: /data/simulators/ue
    launch_script: CitySample.sh
    port: 9000
    cameras: [{name: front, width: 896, height: 896}]
    camera_ids: {front: 1}
    update_ini_port: true
    spawn_cameras: true
```

UnrealCV controls cameras directly in centimetres and returns RGB frames.
Vanilla UnrealCV does not define a portable collision command. If a packaged
scene provides one, configure `collision_query`; otherwise collision metadata
contains `available: false`. Do not report collision-based benchmark results
for such a scene until a native query or geometry collision plugin is enabled.

## Coordinate contract

AeroLoop always exposes metres, radians, world pose, start-relative state, and
body-frame actions. `position_sign`, `position_scale`, `yaw_sign`, and
`yaw_offset` are the explicit boundary to a simulator's native coordinates.
They must be validated once per scene family with a known pose and camera.
