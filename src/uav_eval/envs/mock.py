from __future__ import annotations

from dataclasses import dataclass

from ..geometry import apply_body_action, distance, relative_state
from ..protocols import EnvironmentAdapter
from ..types import CanonicalAction, EpisodeSpec, Observation, Pose, Transition


@dataclass
class MockEnvironment(EnvironmentAdapter):
    """Dependency-free kinematic environment for tests and integration smoke runs."""

    collision_x: float | None = None
    name: str = "mock"

    def reset(self, episode: EpisodeSpec) -> Observation:
        self.episode = episode
        self.origin = episode.start_pose
        self.pose = episode.start_pose
        self.step = 0
        return self._observation()

    def _observation(self) -> Observation:
        return Observation(
            rgb=None,
            pose=self.pose,
            relative_state=relative_state(self.pose, self.origin),
            step_index=self.step,
            info={"mock": True},
        )

    def execute(self, action: CanonicalAction) -> Transition:
        self.pose = apply_body_action(self.pose, action)
        self.step += 1
        collision = self.collision_x is not None and self.pose.x >= self.collision_x
        surface_distance = distance(self.pose.xyz(), self.episode.target_position, dimensions=3)
        return Transition(self._observation(), collision=collision, info={"surface_distance": surface_distance})
