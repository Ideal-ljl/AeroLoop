from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class CameraSpec:
    """A camera pose relative to the UAV body frame.

    Position is expressed in benchmark distance units. Orientation is stored in
    radians. ``width``, ``height`` and ``fov`` are requests; an environment may
    reject unsupported values instead of silently changing them.
    """

    name: str
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    width: int | None = None
    height: int | None = None
    fov: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("camera name cannot be empty")
        position = tuple(map(float, self.position))
        if len(position) != 3 or not all(math.isfinite(value) for value in position):
            raise ValueError("camera position must contain three finite values")
        angles = tuple(map(float, (self.yaw, self.pitch, self.roll)))
        if not all(math.isfinite(value) for value in angles):
            raise ValueError("camera orientation must be finite")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "yaw", angles[0])
        object.__setattr__(self, "pitch", angles[1])
        object.__setattr__(self, "roll", angles[2])
        for field_name in ("width", "height"):
            value = getattr(self, field_name)
            if value is not None and int(value) <= 0:
                raise ValueError(f"camera {field_name} must be positive")
        if self.fov is not None and not 0.0 < float(self.fov) < 180.0:
            raise ValueError("camera fov must be between 0 and 180 degrees")
        object.__setattr__(self, "width", int(self.width) if self.width is not None else None)
        object.__setattr__(self, "height", int(self.height) if self.height is not None else None)
        object.__setattr__(self, "fov", float(self.fov) if self.fov is not None else None)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "CameraSpec":
        def angle(name: str) -> float:
            if f"{name}_degrees" in row:
                return math.radians(float(row[f"{name}_degrees"]))
            return float(row.get(name, 0.0))

        return cls(
            name=str(row["name"]),
            position=tuple(map(float, row.get("position", (0.0, 0.0, 0.0)))),
            yaw=angle("yaw"),
            pitch=angle("pitch"),
            roll=angle("roll"),
            width=int(row["width"]) if row.get("width") is not None else None,
            height=int(row["height"]) if row.get("height") is not None else None,
            fov=float(row["fov"]) if row.get("fov") is not None else None,
            metadata=dict(row.get("metadata", {})),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "position": list(self.position),
            "yaw": self.yaw,
            "pitch": self.pitch,
            "roll": self.roll,
            "width": self.width,
            "height": self.height,
            "fov": self.fov,
            "metadata": dict(self.metadata),
        }


STANDARD_CAMERAS: Mapping[str, CameraSpec] = {
    "front": CameraSpec("front"),
    "back": CameraSpec("back", yaw=math.pi),
    "left": CameraSpec("left", yaw=math.pi / 2.0),
    "right": CameraSpec("right", yaw=-math.pi / 2.0),
    "down": CameraSpec("down", pitch=-math.pi / 2.0),
}


def resolve_cameras(value: str | Iterable[str | Mapping[str, Any] | CameraSpec] | None) -> tuple[CameraSpec, ...]:
    """Resolve a camera config.

    Presets are ``front`` (the compatibility default) and ``standard`` (front,
    back, left, right, down). Lists may mix preset names and custom mappings.
    """

    if value is None or value == "front":
        return (STANDARD_CAMERAS["front"],)
    if value == "standard":
        return tuple(STANDARD_CAMERAS.values())
    if isinstance(value, (str, Mapping)):
        value = [value]
    cameras: list[CameraSpec] = []
    for item in value:
        if isinstance(item, CameraSpec):
            camera = item
        elif isinstance(item, str):
            try:
                camera = STANDARD_CAMERAS[item]
            except KeyError as exc:
                raise KeyError(f"unknown camera preset: {item}") from exc
        else:
            camera = CameraSpec.from_mapping(item)
        if any(existing.name == camera.name for existing in cameras):
            raise ValueError(f"duplicate camera name: {camera.name}")
        cameras.append(camera)
    if not cameras:
        raise ValueError("at least one camera is required")
    return tuple(cameras)
