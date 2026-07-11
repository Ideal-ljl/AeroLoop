from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Sequence
from urllib import request

from ..protocols import PolicyAdapter
from ..types import ActionChunk, EpisodeSpec, PolicyInput


def _encode_rgb_png(image) -> str | None:
    if image is None:
        return None
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("HttpPolicy image encoding requires opencv-python") from exc
    bgr = image[..., ::-1]
    ok, encoded = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("failed to encode observation as PNG")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


@dataclass
class HttpPolicy(PolicyAdapter):
    """Canonical HTTP policy client, suitable for model-specific conda environments.

    Predict request::
      {episode_id, instruction, step, state, image_base64, image_format}

    Predict response::
      {actions: [[dx,dy,dz,stop] | [dx,dy,dz,d_yaw,stop], ...], metadata: {...}}
    """

    url: str
    timeout_s: float = 120.0
    reset_url: str | None = None
    views: Sequence[str] | None = None
    primary_view: str = "front"
    name: str = "http"

    def __post_init__(self) -> None:
        if isinstance(self.views, str):
            self.views = (self.views,)
        elif self.views is not None:
            self.views = tuple(str(view) for view in self.views)
        self.primary_view = str(self.primary_view)

    def _post(self, url: str, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        with request.urlopen(req, timeout=self.timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    def reset(self, episode: EpisodeSpec) -> None:
        self.episode = episode
        if self.reset_url:
            self._post(
                self.reset_url,
                {"episode_id": episode.episode_id, "instruction": episode.instruction, "env_name": episode.env_name},
            )

    def predict(self, policy_input: PolicyInput) -> ActionChunk:
        obs = policy_input.observation
        available = dict(obs.images)
        if not available and obs.rgb is not None:
            available[obs.primary_view] = obs.rgb
        requested_views = self.views if self.views is not None else tuple(available)
        missing = [view for view in requested_views if view not in available]
        if missing:
            raise ValueError(f"policy requires unavailable views {missing}; available views: {sorted(available)}")
        encoded_views = {view: _encode_rgb_png(available[view]) for view in requested_views}
        primary_view = self.primary_view if self.primary_view in encoded_views else obs.primary_view
        if primary_view not in encoded_views and encoded_views:
            primary_view = next(iter(encoded_views))
        camera_specs = {view: obs.camera_specs[view].as_dict() for view in requested_views if view in obs.camera_specs}
        response = self._post(
            self.url,
            {
                "episode_id": policy_input.episode.episode_id,
                "env_name": policy_input.episode.env_name,
                "instruction": policy_input.episode.instruction,
                "step": obs.step_index,
                "state": list(obs.relative_state),
                "pose": obs.pose.as_list(),
                "auxiliary_state": dict(obs.auxiliary_state),
                "image_base64": encoded_views.get(primary_view),
                "image_format": "png_rgb",
                "images_base64": encoded_views,
                "primary_view": primary_view,
                "camera_specs": camera_specs,
            },
        )
        rows = response.get("actions")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"HTTP policy returned no actions: {response}")
        return ActionChunk.from_rows(rows, response.get("metadata") or {})
