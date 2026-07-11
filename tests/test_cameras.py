import math
import unittest

from uav_eval.cameras import CameraSpec, resolve_cameras
from uav_eval.envs.airbrain import AirBrainEnvironment
from uav_eval.envs.mock import MockEnvironment
from uav_eval.types import EpisodeSpec, Observation, Pose


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
        environment = MockEnvironment(
            cameras=["front", "down"], image_factory=lambda camera, pose, step: f"{camera.name}:{step}"
        )
        episode = EpisodeSpec("ep", "mock", "go", Pose(0, 0, 0, 0), (1, 0, 0), 1)
        observation = environment.reset(episode)
        self.assertEqual(observation.images, {"front": "front:0", "down": "down:0"})
        self.assertEqual(observation.rgb, "front:0")

    def test_airbrain_applies_body_relative_camera_pose(self):
        calls = []

        class Bridge:
            def set_camera_pose(self, *values):
                calls.append(values)

        environment = AirBrainEnvironment.__new__(AirBrainEnvironment)
        environment.pose = Pose(10, 20, 30, math.pi / 2)
        environment.episode = type("Episode", (), {"metadata": {"pitch": -10}})()
        environment.pos_ratio = 2.0
        environment._bridge = Bridge()
        environment.settle_seconds = 0
        environment._set_pose(CameraSpec("offset", position=(2, 0, 4), yaw=math.pi / 2, pitch=-math.pi / 4))
        x, y, z, pitch, yaw, roll = calls[0]
        self.assertAlmostEqual(x, 5)
        self.assertAlmostEqual(y, 11)
        self.assertAlmostEqual(z, 17)
        self.assertAlmostEqual(pitch, -55)
        self.assertAlmostEqual(yaw, 180)
        self.assertAlmostEqual(roll, 0)


if __name__ == "__main__":
    unittest.main()
