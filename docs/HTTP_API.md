# Model HTTP API

Every model service exposes `GET /health`, `POST /v1/reset`, and
`POST /v1/predict`. A model service converts its native inputs and outputs; the
benchmark runner never imports the model package.

## Reset

```json
{
  "episode_id": "env:episode",
  "env_name": "env_airsim_18",
  "instruction": "fly toward the tower"
}
```

## Predict

```json
{
  "episode_id": "env:episode",
  "env_name": "env_airsim_18",
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
