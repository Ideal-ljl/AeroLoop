from __future__ import annotations

import base64
import io
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping


def heading_delta_from_translation(dx: float, dy: float, epsilon: float = 1e-8) -> float:
    """Match AirBrain's local-trajectory convention with an explicit yaw delta."""
    if math.hypot(float(dx), float(dy)) <= epsilon:
        return 0.0
    return math.atan2(float(dy), float(dx))


@dataclass(frozen=True)
class PredictRequest:
    episode_id: str
    env_name: str
    instruction: str
    step: int
    state: tuple[float, float, float, float]
    pose: tuple[float, float, float, float]
    image_base64: str | None
    image_format: str = "png_rgb"

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "PredictRequest":
        state = tuple(map(float, row.get("state", ())))
        pose = tuple(map(float, row.get("pose", ())))
        if len(state) != 4 or len(pose) != 4:
            raise ValueError("state and pose must each contain [x,y,z,yaw]")
        if not all(math.isfinite(x) for x in (*state, *pose)):
            raise ValueError("state and pose must be finite")
        episode_id = str(row.get("episode_id", "")).strip()
        if not episode_id:
            raise ValueError("episode_id is required")
        return cls(
            episode_id=episode_id,
            env_name=str(row.get("env_name", "")),
            instruction=str(row.get("instruction", "")),
            step=int(row.get("step", 0)),
            state=state,
            pose=pose,
            image_base64=row.get("image_base64"),
            image_format=str(row.get("image_format", "png_rgb")),
        )

    def decode_rgb(self):
        if not self.image_base64:
            raise ValueError("image_base64 is required by this backend")
        try:
            import numpy as np
            from PIL import Image
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("image decoding requires NumPy and Pillow in the model environment") from exc
        raw = base64.b64decode(self.image_base64, validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)


class ModelBackend(ABC):
    name = "model"

    def health(self) -> Mapping[str, Any]:
        return {"status": "ok", "backend": self.name}

    @abstractmethod
    def reset(self, episode_id: str, instruction: str, env_name: str) -> Mapping[str, Any] | None:
        pass

    @abstractmethod
    def predict(self, request: PredictRequest) -> Mapping[str, Any]:
        """Return {actions: [[...], ...], metadata: {...}} in canonical units."""

    def close(self) -> None:
        pass
