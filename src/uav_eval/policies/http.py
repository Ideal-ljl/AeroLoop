from __future__ import annotations

import base64
import json
from dataclasses import dataclass
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
    name: str = "http"

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
        response = self._post(
            self.url,
            {
                "episode_id": policy_input.episode.episode_id,
                "env_name": policy_input.episode.env_name,
                "instruction": policy_input.episode.instruction,
                "step": obs.step_index,
                "state": list(obs.relative_state),
                "pose": obs.pose.as_list(),
                "image_base64": _encode_rgb_png(obs.rgb),
                "image_format": "png_rgb",
            },
        )
        rows = response.get("actions")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"HTTP policy returned no actions: {response}")
        return ActionChunk.from_rows(rows, response.get("metadata") or {})
