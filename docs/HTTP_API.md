# Model HTTP API

Every model service exposes `GET /health`, `POST /v1/reset`, and
`POST /v1/predict`. A model service converts its native inputs and outputs; the
benchmark runner never imports the model package.

This boundary deliberately allows the simulator/runner and model to use
different Python and Conda environments. They only need compatible protocol
versions and network access; CUDA, PyTorch, simulator SDK, and model
dependencies do not cross the boundary.

## Reset

```json
{
  "episode_id": "scene:episode",
  "env_name": "custom-simulator",
  "instruction": "fly toward the tower"
}
```

## Predict

```json
{
  "episode_id": "scene:episode",
  "env_name": "custom-simulator",
  "instruction": "fly toward the tower",
  "step": 0,
  "state": [0.0, 0.0, 0.0, 0.0],
  "pose": [12.0, 8.0, 5.0, 1.57],
  "auxiliary_state": {"velocity": [0.1, 0.0, 0.0]},
  "primary_view": "front",
  "images_base64": {
    "front": "<PNG>",
    "down": "<PNG>"
  },
  "camera_specs": {
    "front": {
      "position": [0.0, 0.0, 0.0],
      "yaw": 0.0,
      "pitch": 0.0,
      "roll": 0.0,
      "width": 640,
      "height": 480,
      "fov": 90.0
    }
  },
  "image_base64": "<front PNG>",
  "image_format": "png_rgb"
}
```

`image_base64` is the compatibility alias for the primary view. New adapters
should read `images_base64`; `PredictRequest.decode_rgb("down")` decodes a
named view. `camera_specs` describes body-relative extrinsics in radians and
the renderer's actual image dimensions/FOV when available.

The response contains one or more canonical actions:

```json
{
  "actions": [
    [0.5, 0.0, 0.0, 0.1, 0.0],
    [0.5, 0.0, 0.0, 0.0, 0.0]
  ],
  "metadata": {"model": "example", "native_chunk_size": 2}
}
```

Rows must contain finite numeric values. Four-column rows
`[dx,dy,dz,stop]` are accepted and imply `d_yaw=0`.

## Payload selection and inference-function wrapper

```yaml
policy:
  type: http
  kwargs:
    views: [front]
    observation_fields: [images, state]
    state_source: relative
    image_size: [224, 224]
    view_sizes:
      down: [320, 240]
```

Available fields are `images`, `state` (start-frame-relative), `pose` (world
frame), `auxiliary_state`, and `camera_specs`. Episode identity, instruction,
environment name, and step index are always sent.

`state_source` may be `relative`, `world`, or an integration-provided selector
such as `auxiliary.body_state` or `auxiliary.ned_state`. Canonical actions are
always body-frame translations; simulator integrations own conversion to their
native ENU/NED/world coordinates.

`aeroloop-model-server function` maps arbitrary inference-function argument names
to selectors such as `instruction`, `state`, `pose`,
`auxiliary_state.velocity`, `image`, or `images.down`. The function returns an
action list or an object containing `actions` and optional `metadata`.
