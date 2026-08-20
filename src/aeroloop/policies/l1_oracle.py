from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from ..geometry import wrap_angle
from ..protocols import PolicyAdapter
from ..types import ActionChunk, CanonicalAction, EpisodeSpec, PolicyInput, Pose, Transition


@dataclass(frozen=True)
class L1Guidance:
    stage: str
    reference_index: int
    reference_point: tuple[float, float, float]
    heading_error_degrees: float
    distance_to_end: float


def l1_oracle_guidance(
    trajectory: Sequence[Sequence[float]],
    pose: Pose,
    step_index: int,
    *,
    lookahead_distance: float = 3.0,
    turn_threshold_degrees: float = 20.0,
    takeoff_height: float = 3.0,
    landing_height: float = 7.0,
    landing_radius: float = 10.0,
    takeoff_steps: int = 6,
) -> L1Guidance:
    """Return OpenUAV L1's high-level oracle stage in canonical coordinates."""

    points = [tuple(map(float, row[:3])) for row in trajectory if len(row) >= 3]
    if not points:
        raise ValueError("L1 oracle assistance requires a non-empty 3D reference trajectory")
    if not all(math.isfinite(value) for point in points for value in point):
        raise ValueError("L1 oracle trajectory must contain only finite coordinates")

    nearest_index = min(
        range(len(points)),
        key=lambda index: (points[index][0] - pose.x) ** 2 + (points[index][1] - pose.y) ** 2,
    )
    reference_index = len(points) - 1
    for index in range(nearest_index, len(points)):
        horizontal_distance = math.hypot(points[index][0] - pose.x, points[index][1] - pose.y)
        if horizontal_distance > float(lookahead_distance) or index == len(points) - 1:
            reference_index = index
            break

    reference = points[reference_index]
    dx, dy, dz = reference[0] - pose.x, reference[1] - pose.y, reference[2] - pose.z
    horizontal_distance = math.hypot(dx, dy)
    end = points[-1]
    distance_to_end = math.hypot(end[0] - pose.x, end[1] - pose.y)
    heading_error = 0.0 if horizontal_distance <= 1e-8 else wrap_angle(math.atan2(dy, dx) - pose.yaw)

    if int(step_index) < int(takeoff_steps):
        stage = "take off"
    elif dz > float(takeoff_height) and abs(dz) >= horizontal_distance:
        stage = "take off"
    elif distance_to_end <= float(landing_radius) or dz < -float(landing_height):
        stage = "landing"
    elif abs(math.degrees(heading_error)) > float(turn_threshold_degrees):
        stage = "left" if heading_error > 0 else "right"
    else:
        stage = "cruise"

    return L1Guidance(
        stage=stage,
        reference_index=reference_index,
        reference_point=reference,
        heading_error_degrees=math.degrees(heading_error),
        distance_to_end=distance_to_end,
    )


class L1OraclePolicy(PolicyAdapter):
    """Inject OpenUAV L1 GT-trajectory guidance into a wrapped policy."""

    name = "l1_oracle"

    def __init__(
        self,
        base_policy: PolicyAdapter,
        trajectory_metadata_key: str = "trajectory",
        lookahead_distance: float = 3.0,
        turn_threshold_degrees: float = 20.0,
        takeoff_height: float = 3.0,
        landing_height: float = 7.0,
        landing_radius: float = 10.0,
        takeoff_steps: int = 6,
    ):
        self.base_policy = base_policy
        self.trajectory_metadata_key = str(trajectory_metadata_key)
        self.guidance_kwargs: Mapping[str, Any] = {
            "lookahead_distance": float(lookahead_distance),
            "turn_threshold_degrees": float(turn_threshold_degrees),
            "takeoff_height": float(takeoff_height),
            "landing_height": float(landing_height),
            "landing_radius": float(landing_radius),
            "takeoff_steps": int(takeoff_steps),
        }

    def reset(self, episode: EpisodeSpec) -> None:
        self.base_policy.reset(episode)

    def predict(self, policy_input: PolicyInput) -> ActionChunk:
        trajectory = policy_input.episode.metadata.get(self.trajectory_metadata_key) or ()
        guidance = l1_oracle_guidance(
            trajectory,
            policy_input.observation.pose,
            policy_input.observation.step_index,
            **self.guidance_kwargs,
        )
        context = {
            **dict(policy_input.policy_context),
            "assistant_level": "L1",
            "assistant_stage": guidance.stage,
        }
        chunk = self.base_policy.predict(replace(policy_input, policy_context=context))
        return ActionChunk(
            chunk.actions,
            {
                **dict(chunk.metadata),
                "assistant_level": "L1",
                "assistant_stage": guidance.stage,
                "assistant_reference_index": guidance.reference_index,
                "assistant_reference_point": list(guidance.reference_point),
                "assistant_heading_error_degrees": guidance.heading_error_degrees,
                "assistant_distance_to_end": guidance.distance_to_end,
            },
        )

    def on_action_executed(self, action: CanonicalAction, transition: Transition) -> None:
        self.base_policy.on_action_executed(action, transition)

    def close(self) -> None:
        self.base_policy.close()
