from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Mapping, Sequence
from urllib import request

from ..protocols import PolicyAdapter
from ..types import ActionChunk, EpisodeSpec, PolicyInput


def _encode_rgb_png(image, size: Sequence[int] | None = None) -> str | None:
    if image is None:
        return None
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("HttpPolicy image encoding requires opencv-python") from exc
    if size is not None:
        if len(size) != 2 or int(size[0]) <= 0 or int(size[1]) <= 0:
            raise ValueError("image_size must be [width, height]")
        image = cv2.resize(image, (int(size[0]), int(size[1])), interpolation=cv2.INTER_AREA)
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
    observation_fields: Sequence[str] | None = None
    state_source: str = "relative"
    image_size: Sequence[int] | None = None
    view_sizes: Mapping[str, Sequence[int]] | None = None
    history_steps: int = 1
    name: str = "http"

    def __post_init__(self) -> None:
        if isinstance(self.views, str):
            self.views = (self.views,)
        elif self.views is not None:
            self.views = tuple(str(view) for view in self.views)
        self.primary_view = str(self.primary_view)
        self.state_source = str(self.state_source)
        if self.state_source not in {"relative", "world"} and not self.state_source.startswith("auxiliary."):
            raise ValueError("state_source must be relative, world, or auxiliary.<key>")
        valid_fields = {"images", "state", "pose", "auxiliary_state", "camera_specs"}
        if self.observation_fields is None:
            self.observation_fields = tuple(sorted(valid_fields))
        else:
            self.observation_fields = tuple(str(field) for field in self.observation_fields)
            unknown = set(self.observation_fields) - valid_fields
            if unknown:
                raise ValueError(f"unknown observation_fields: {sorted(unknown)}")
        if self.image_size is not None:
            self.image_size = tuple(map(int, self.image_size))
        self.view_sizes = {str(name): tuple(map(int, size)) for name, size in (self.view_sizes or {}).items()}
        self.history_steps = int(self.history_steps)
        if self.history_steps < 1:
            raise ValueError("history_steps must be at least 1")

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
        include_images = "images" in self.observation_fields
        encoded_views = {}
        encoded_history = {}
        if include_images:
            for view in requested_views:
                size = self.view_sizes.get(view, self.image_size)
                encoded_views[view] = (
                    _encode_rgb_png(available[view], size) if size is not None else _encode_rgb_png(available[view])
                )
                frames = []
                if self.history_steps > 1:
                    for historical_views in policy_input.view_history[-(self.history_steps - 1) :]:
                        historical = historical_views.get(view)
                        if historical is not None:
                            frames.append(historical)
                frames.append(available[view])
                while len(frames) < self.history_steps:
                    frames.insert(0, frames[0])
                encoded_history[view] = [
                    _encode_rgb_png(frame, size) if size is not None else _encode_rgb_png(frame)
                    for frame in frames[-self.history_steps :]
                ]
        primary_view = self.primary_view if self.primary_view in encoded_views else obs.primary_view
        if primary_view not in encoded_views and encoded_views:
            primary_view = next(iter(encoded_views))
        camera_specs = {}
        if "camera_specs" in self.observation_fields:
            camera_specs = {
                view: obs.camera_specs[view].as_dict() for view in requested_views if view in obs.camera_specs
            }
            for view, spec in camera_specs.items():
                size = self.view_sizes.get(view, self.image_size)
                if size is not None:
                    spec.update({"width": int(size[0]), "height": int(size[1])})
        payload = {
            "episode_id": policy_input.episode.episode_id,
            "env_name": policy_input.episode.env_name,
            "instruction": policy_input.episode.instruction,
            "step": obs.step_index,
            "policy_context": dict(policy_input.policy_context),
        }
        if "state" in self.observation_fields:
            if self.state_source == "relative":
                state = obs.relative_state
            elif self.state_source == "world":
                state = obs.pose.as_list()
            else:
                state = obs.auxiliary_state
                for part in self.state_source.split(".")[1:]:
                    state = state[part]
            payload["state"] = list(state)
        if "pose" in self.observation_fields:
            payload["pose"] = obs.pose.as_list()
        if "auxiliary_state" in self.observation_fields:
            payload["auxiliary_state"] = dict(obs.auxiliary_state)
        if include_images:
            payload.update(
                {
                    "image_base64": encoded_views.get(primary_view),
                    "image_format": "png_rgb",
                    "images_base64": encoded_views,
                    "image_history_base64": encoded_history,
                    "primary_view": primary_view,
                }
            )
        if "camera_specs" in self.observation_fields:
            payload["camera_specs"] = camera_specs
        response = self._post(
            self.url,
            payload,
        )
        rows = response.get("actions")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"HTTP policy returned no actions: {response}")
        return ActionChunk.from_rows(rows, response.get("metadata") or {})
