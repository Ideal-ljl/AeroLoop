import math
import unittest

from aeroloop.cameras import resolve_cameras
from aeroloop.simulators.mock import MockSimulator
from aeroloop.types import EpisodeSpec, Observation, Pose


class CameraTest(unittest.TestCase):
    def test_standard_preset_has_five_named_views(self):
        cameras = resolve_cameras("standard")
        self.assertEqual([camera.name for camera in cameras], ["front", "back", "left", "right", "down"])
        self.assertAlmostEqual(cameras[1].yaw, math.pi)
        self.assertAlmostEqual(cameras[-1].pitch, -math.pi / 2)

    def test_custom_camera_accepts_degree_config(self):
        cameras = resolve_cameras([{"name": "gimbal", "position": [1, 2, 3], "yaw_degrees": 45}])
        self.assertEqual(cameras[0].position, (1.0, 2.0, 3.0))
        self.assertAlmostEqual(cameras[0].yaw, math.pi / 4)

    def test_observation_keeps_legacy_rgb_and_named_views_in_sync(self):
        front = object()
        observation = Observation(front, Pose(0, 0, 0, 0), (0, 0, 0, 0), 0)
        self.assertIs(observation.image("front"), front)
        self.assertEqual(observation.available_views, ("front",))

        down = object()
        observation = Observation(None, Pose(0, 0, 0, 0), (0, 0, 0, 0), 0, images={"down": down}, primary_view="down")
        self.assertIs(observation.rgb, down)

    def test_mock_environment_can_render_configured_views(self):
        environment = MockSimulator(
            cameras=["front", "down"], image_factory=lambda camera, pose, step: f"{camera.name}:{step}"
        )
        episode = EpisodeSpec("ep", "mock", "go", Pose(0, 0, 0, 0), (1, 0, 0), 1)
        observation = environment.reset(episode)
        self.assertEqual(observation.images, {"front": "front:0", "down": "down:0"})
        self.assertEqual(observation.rgb, "front:0")


if __name__ == "__main__":
    unittest.main()
