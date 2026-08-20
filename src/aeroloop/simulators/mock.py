from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..cameras import CameraSpec, resolve_cameras
from ..geometry import apply_body_action, relative_state
from ..protocols import SimulatorAdapter
from ..types import CanonicalAction, EpisodeSpec, Observation, Pose, Transition


@dataclass
class MockSimulator(SimulatorAdapter):
    """Dependency-free kinematic simulator used only for tests and smoke runs."""

    collision_x: float | None = None
    cameras: Any = "front"
    image_factory: Callable[[CameraSpec, Pose, int], Any] | None = None
    name: str = "mock"
    camera_specs: tuple[CameraSpec, ...] = field(init=False)

    def __post_init__(self) -> None:
        self.camera_specs = resolve_cameras(self.cameras)

    def reset(self, episode: EpisodeSpec) -> Observation:
        self.episode = episode
        self.origin = episode.start_pose
        self.pose = episode.start_pose
        self.step = 0
        return self._observation()

    def _observation(self) -> Observation:
        images = {}
        if self.image_factory is not None:
            images = {camera.name: self.image_factory(camera, self.pose, self.step) for camera in self.camera_specs}
        return Observation(
            rgb=images.get(self.camera_specs[0].name),
            pose=self.pose,
            relative_state=relative_state(self.pose, self.origin),
            step_index=self.step,
            info={"mock": True},
            images=images,
            primary_view=self.camera_specs[0].name,
            camera_specs={camera.name: camera for camera in self.camera_specs},
        )

    def execute(self, action: CanonicalAction) -> Transition:
        self.pose = apply_body_action(self.pose, action)
        self.step += 1
        collision = self.collision_x is not None and self.pose.x >= self.collision_x
        return Transition(self._observation(), collision=collision, info={})
