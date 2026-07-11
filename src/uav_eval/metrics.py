from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping, Sequence

from .geometry import distance
from .types import CanonicalAction, EpisodeSpec, Pose, TerminationReason


class Metric(ABC):
    """Lifecycle for a user-defined per-episode metric.

    Metric output is namespaced as ``<name>/<key>`` so third-party metrics
    cannot accidentally overwrite benchmark metrics.
    """

    name = "custom"

    @abstractmethod
    def reset(self, episode: EpisodeSpec) -> None:
        pass

    def update(
        self,
        before: Pose,
        after: Pose,
        *,
        action: CanonicalAction,
        collision: bool,
        info: Mapping[str, Any],
    ) -> None:
        pass

    @abstractmethod
    def finalize(self, termination: TerminationReason, final_pose: Pose, step_count: int) -> Mapping[str, Any]:
        pass


@dataclass(frozen=True)
class MetricConfig:
    success_distance: float = 25.0
    distance_mode: str = "legacy_min"
    require_stop_for_success: bool = False

    def __post_init__(self) -> None:
        valid = {"legacy_min", "endpoint_2d", "endpoint_3d", "surface"}
        if self.distance_mode not in valid:
            raise ValueError(f"distance_mode must be one of {sorted(valid)}")
        if self.success_distance <= 0:
            raise ValueError("success_distance must be positive")


def compute_distances(episode: EpisodeSpec, pose: Pose, info: Mapping[str, Any]) -> dict[str, float]:
    endpoint_2d = distance(pose.xyz(), episode.target_position, dimensions=2)
    endpoint_3d = distance(pose.xyz(), episode.target_position, dimensions=3)
    building_2d = float("inf")
    if episode.building_position is not None:
        building_2d = distance(pose.xyz(), episode.building_position, dimensions=2)
    surface = float(info.get("surface_distance", float("inf")))
    return {
        "endpoint_2d": endpoint_2d,
        "endpoint_3d": endpoint_3d,
        "building_2d": building_2d,
        "surface": surface,
        "legacy_min": min(endpoint_2d, building_2d, surface),
    }


class EpisodeMetrics:
    def __init__(self, episode: EpisodeSpec, config: MetricConfig):
        self.episode = episode
        self.config = config
        self.path_length = 0.0
        self.collision = False
        self.oracle_success = False
        self.min_distance = float("inf")
        self.last_distances: dict[str, float] = {}
        self.inference_calls = 0
        self.inference_ms = 0.0

    def update(
        self,
        before: Pose,
        after: Pose,
        *,
        collision: bool,
        info: Mapping[str, Any],
    ) -> dict[str, float]:
        self.path_length += distance(before.xyz(), after.xyz(), dimensions=3)
        self.collision = self.collision or collision
        self.last_distances = compute_distances(self.episode, after, info)
        selected = self.last_distances[self.config.distance_mode]
        self.min_distance = min(self.min_distance, selected)
        self.oracle_success = self.oracle_success or selected < self.config.success_distance
        return self.last_distances

    def add_inference(self, elapsed_ms: float) -> None:
        self.inference_calls += 1
        self.inference_ms += float(elapsed_ms)

    def finalize(self, termination: TerminationReason, final_pose: Pose, step_count: int) -> dict[str, Any]:
        if not self.last_distances:
            self.last_distances = compute_distances(self.episode, final_pose, {})
            self.min_distance = self.last_distances[self.config.distance_mode]
        stopped = termination == TerminationReason.STOP
        in_goal = self.last_distances[self.config.distance_mode] < self.config.success_distance
        success = in_goal and not self.collision and (stopped or not self.config.require_stop_for_success)
        ref = max(float(self.episode.reference_path_length), 0.0)
        spl = min(1.0, ref / max(self.path_length, ref, 1e-8)) if success else 0.0
        return {
            "success": int(success),
            "spl": float(spl),
            "osr": int(self.oracle_success and not self.collision),
            "collision": self.collision,
            "stop_called": stopped,
            "stop_success": bool(stopped and in_goal and not self.collision),
            "premature_stop": bool(stopped and not in_goal),
            "timeout": termination == TerminationReason.MAX_STEPS,
            "steps_taken": int(step_count),
            "path_length": float(self.path_length),
            "reference_path_length": ref,
            "distance_to_goal": float(self.last_distances[self.config.distance_mode]),
            "min_distance_to_goal": float(self.min_distance),
            **{f"final_{key}": float(value) for key, value in self.last_distances.items()},
            "inference_calls": self.inference_calls,
            "total_inference_ms": self.inference_ms,
            "avg_inference_ms": self.inference_ms / self.inference_calls if self.inference_calls else 0.0,
        }


class MetricSuite:
    """Built-in benchmark metrics plus isolated third-party metrics."""

    def __init__(self, episode: EpisodeSpec, config: MetricConfig, custom: Sequence[Metric] = ()):
        self.episode = episode
        self.builtin = EpisodeMetrics(episode, config)
        self.custom = tuple(custom)
        names = [metric.name for metric in self.custom]
        if any(not str(name).strip() for name in names):
            raise ValueError("custom metric names cannot be empty")
        if len(set(names)) != len(names):
            raise ValueError(f"custom metric names must be unique, got {names}")

    def reset(self) -> None:
        for metric in self.custom:
            metric.reset(self.episode)

    def update(
        self,
        before: Pose,
        after: Pose,
        *,
        action: CanonicalAction,
        collision: bool,
        info: Mapping[str, Any],
    ) -> dict[str, float]:
        distances = self.builtin.update(before, after, collision=collision, info=info)
        for metric in self.custom:
            metric.update(before, after, action=action, collision=collision, info=info)
        return distances

    def add_inference(self, elapsed_ms: float) -> None:
        self.builtin.add_inference(elapsed_ms)

    def finalize(self, termination: TerminationReason, final_pose: Pose, step_count: int) -> dict[str, Any]:
        values = self.builtin.finalize(termination, final_pose, step_count)
        for metric in self.custom:
            output = metric.finalize(termination, final_pose, step_count)
            if not isinstance(output, Mapping):
                raise TypeError(f"metric {metric.name!r} finalize() must return a mapping")
            for key, value in output.items():
                values[f"{metric.name}/{key}"] = value
        return values

    def finalize_builtin(self, termination: TerminationReason, final_pose: Pose, step_count: int) -> dict[str, Any]:
        return self.builtin.finalize(termination, final_pose, step_count)


def aggregate_results(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"type": "summary", "total_samples": 0}
    means = ("success", "spl", "osr", "collision", "stop_called", "stop_success", "premature_stop", "distance_to_goal")
    summary: dict[str, Any] = {"type": "summary", "total_samples": len(rows)}
    for key in means:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        summary[key] = sum(values) / len(values) if values else 0.0
    custom_keys = sorted({key for row in rows for key in row if "/" in key})
    for key in custom_keys:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), Real) and math.isfinite(float(row[key]))]
        if values:
            summary[key] = sum(values) / len(values)
    summary["total_steps"] = sum(int(row.get("steps_taken", 0)) for row in rows)
    summary["total_inference_calls"] = sum(int(row.get("inference_calls", 0)) for row in rows)
    return summary
