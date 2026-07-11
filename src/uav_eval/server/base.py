from __future__ import annotations

import base64
import io
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
    images_base64: Mapping[str, str | None] = field(default_factory=dict)
    primary_view: str = "front"
    auxiliary_state: Mapping[str, Any] = field(default_factory=dict)
    camera_specs: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        images = dict(self.images_base64 or {})
        if self.image_base64 is not None:
            images.setdefault(self.primary_view, self.image_base64)
        object.__setattr__(self, "images_base64", images)

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
        raw_images = row.get("images_base64") or {}
        if not isinstance(raw_images, Mapping):
            raise ValueError("images_base64 must be an object keyed by view name")
        images = {}
        for name, value in raw_images.items():
            if value is not None and not isinstance(value, str):
                raise ValueError(f"image for view {name!r} must be a base64 string or null")
            images[str(name)] = value
        auxiliary_state = row.get("auxiliary_state") or {}
        if not isinstance(auxiliary_state, Mapping):
            raise ValueError("auxiliary_state must be an object")
        raw_camera_specs = row.get("camera_specs") or {}
        if not isinstance(raw_camera_specs, Mapping):
            raise ValueError("camera_specs must be an object keyed by view name")
        camera_specs = {}
        for name, spec in raw_camera_specs.items():
            if not isinstance(spec, Mapping):
                raise ValueError(f"camera spec for view {name!r} must be an object")
            camera_specs[str(name)] = dict(spec)
        return cls(
            episode_id=episode_id,
            env_name=str(row.get("env_name", "")),
            instruction=str(row.get("instruction", "")),
            step=int(row.get("step", 0)),
            state=state,
            pose=pose,
            image_base64=row.get("image_base64"),
            image_format=str(row.get("image_format", "png_rgb")),
            images_base64=images,
            primary_view=str(row.get("primary_view", "front")),
            auxiliary_state=dict(auxiliary_state),
            camera_specs=camera_specs,
        )

    @property
    def available_views(self) -> tuple[str, ...]:
        return tuple(self.images_base64)

    def decode_rgb(self, view: str | None = None):
        selected_view = view or self.primary_view
        encoded = self.images_base64.get(selected_view)
        if encoded is None and view is None:
            encoded = self.image_base64 or next((value for value in self.images_base64.values() if value), None)
        if not encoded:
            raise ValueError(
                f"image for view {selected_view!r} is required; available views: {sorted(self.images_base64)}"
            )
        try:
            import numpy as np
            from PIL import Image
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("image decoding requires NumPy and Pillow in the model environment") from exc
        raw = base64.b64decode(encoded, validate=True)
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
