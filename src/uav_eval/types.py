from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from .cameras import CameraSpec


def _finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    return value


@dataclass(frozen=True)
class Pose:
    """World-frame pose in benchmark units and radians."""

    x: float
    y: float
    z: float
    yaw: float

    def __post_init__(self) -> None:
        for name in ("x", "y", "z", "yaw"):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))

    @classmethod
    def from_sequence(cls, values: Sequence[float]) -> "Pose":
        if len(values) < 4:
            raise ValueError(f"pose needs [x,y,z,yaw], got {values}")
        return cls(*map(float, values[:4]))

    def xyz(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z

    def as_list(self) -> list[float]:
        return [self.x, self.y, self.z, self.yaw]


@dataclass(frozen=True)
class CanonicalAction:
    """Body-frame translation, explicit yaw delta, and soft stop probability."""

    dx: float
    dy: float
    dz: float
    d_yaw: float = 0.0
    stop: float = 0.0

    def __post_init__(self) -> None:
        for name in ("dx", "dy", "dz", "d_yaw", "stop"):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))

    @classmethod
    def from_sequence(cls, values: Sequence[float]) -> "CanonicalAction":
        values = list(values)
        if len(values) == 4:
            # Compatibility with the OpenFly convention [dx,dy,dz,stop].
            return cls(values[0], values[1], values[2], 0.0, values[3])
        if len(values) >= 5:
            return cls(*values[:5])
        raise ValueError(f"action needs 4 or 5 values, got {values}")

    def zero_motion(self) -> "CanonicalAction":
        return CanonicalAction(0.0, 0.0, 0.0, 0.0, self.stop)

    def as_list(self) -> list[float]:
        return [self.dx, self.dy, self.dz, self.d_yaw, self.stop]


@dataclass(frozen=True)
class ActionChunk:
    actions: tuple[CanonicalAction, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.actions:
            raise ValueError("an action chunk cannot be empty")

    @classmethod
    def from_rows(cls, rows: Iterable[Sequence[float]], metadata: Mapping[str, Any] | None = None) -> "ActionChunk":
        return cls(tuple(CanonicalAction.from_sequence(row) for row in rows), metadata or {})


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: str
    env_name: str
    instruction: str
    start_pose: Pose
    target_position: tuple[float, float, float]
    reference_path_length: float
    building_position: tuple[float, float, float] | None = None
    target_id: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Observation:
    rgb: Any
    pose: Pose
    relative_state: tuple[float, float, float, float]
    step_index: int
    timestamp: float = field(default_factory=time.time)
    info: Mapping[str, Any] = field(default_factory=dict)
    images: Mapping[str, Any] = field(default_factory=dict)
    primary_view: str = "front"
    auxiliary_state: Mapping[str, Any] = field(default_factory=dict)
    camera_specs: Mapping[str, CameraSpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        images = {str(name): image for name, image in self.images.items()}
        rgb = self.rgb
        primary_view = str(self.primary_view)
        if rgb is not None:
            images[primary_view] = rgb
        elif images:
            if primary_view not in images:
                primary_view = next(iter(images))
            rgb = images[primary_view]
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "images", images)
        object.__setattr__(self, "primary_view", primary_view)

    def image(self, view: str | None = None) -> Any:
        """Return a named view, or the primary compatibility RGB image."""

        if view is None:
            return self.rgb
        try:
            return self.images[view]
        except KeyError as exc:
            raise KeyError(f"view {view!r} is unavailable; available views: {sorted(self.images)}") from exc

    @property
    def available_views(self) -> tuple[str, ...]:
        return tuple(self.images)


@dataclass(frozen=True)
class PolicyInput:
    episode: EpisodeSpec
    observation: Observation
    image_history: tuple[Any, ...] = ()
    action_history: tuple[CanonicalAction, ...] = ()
    state_history: tuple[tuple[float, float, float, float], ...] = ()
    view_history: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class Transition:
    observation: Observation
    collision: bool = False
    info: Mapping[str, Any] = field(default_factory=dict)


class TerminationReason(str, Enum):
    STOP = "stop"
    COLLISION = "collision"
    MAX_STEPS = "max_steps"
    ERROR = "error"
    USER_ABORT = "user_abort"


@dataclass(frozen=True)
class StepRecord:
    step: int
    inference_call: int
    action_index: int
    action: CanonicalAction
    pose_before: Pose
    pose_after: Pose
    collision: bool
    inference_ms: float | None
    distances: Mapping[str, float]
    policy_metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "inference_call": self.inference_call,
            "action_index": self.action_index,
            "action": self.action.as_list(),
            "pose_before": self.pose_before.as_list(),
            "pose_after": self.pose_after.as_list(),
            "collision": self.collision,
            "inference_ms": self.inference_ms,
            "distances": dict(self.distances),
            "policy_metadata": dict(self.policy_metadata),
        }


@dataclass(frozen=True)
class EpisodeResult:
    episode_id: str
    env_name: str
    termination_reason: TerminationReason
    metrics: Mapping[str, Any]
    steps: tuple[StepRecord, ...]
    error: str | None = None
    artifacts: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self, include_steps: bool = True) -> dict[str, Any]:
        row = {
            "type": "episode",
            "episode_id": self.episode_id,
            "env_name": self.env_name,
            "termination_reason": self.termination_reason.value,
            **dict(self.metrics),
            "error": self.error,
            "artifacts": dict(self.artifacts),
        }
        if include_steps:
            row["steps"] = [step.as_dict() for step in self.steps]
        return row
