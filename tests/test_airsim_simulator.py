import unittest
from types import SimpleNamespace

from aeroloop.simulators.airsim import AirSimSimulator, GSAirSimSimulator
from aeroloop.types import CanonicalAction, EpisodeSpec, Pose


class FakeVector3r:
    def __init__(self, x, y, z):
        self.x_val, self.y_val, self.z_val = x, y, z


class FakePose:
    def __init__(self, position, orientation):
        self.position = position
        self.orientation = orientation


class FakeImageRequest:
    def __init__(self, camera_name, image_type, pixels_as_float, compress):
        self.camera_name = camera_name
        self.image_type = image_type
        self.pixels_as_float = pixels_as_float


class FakeClient:
    def __init__(self, *args, **kwargs):
        self.poses = []
        self.camera_poses = []
        self.api = []
        self.armed = []
        self.collided = False
        self.collision_timestamp = 0
        self.collide_on_next_pose = False

    def confirmConnection(self):
        pass

    def enableApiControl(self, value, vehicle_name=""):
        self.api.append(value)

    def armDisarm(self, value, vehicle_name=""):
        self.armed.append(value)

    def simSetCameraPose(self, name, pose, vehicle_name=""):
        self.camera_poses.append((name, pose))

    def simSetVehiclePose(self, pose, ignore_collision, vehicle_name=""):
        self.poses.append((pose, ignore_collision))
        if self.collide_on_next_pose:
            self.collided = True
            self.collision_timestamp += 1
            self.collide_on_next_pose = False

    def simGetImages(self, requests, vehicle_name=""):
        responses = []
        for item in requests:
            if item.pixels_as_float:
                responses.append(
                    SimpleNamespace(pixels_as_float=True, width=1, height=1, image_data_float=[12.5])
                )
            else:
                responses.append(
                    SimpleNamespace(
                        pixels_as_float=False,
                        width=1,
                        height=1,
                        image_data_uint8=bytes([10, 20, 30]),
                    )
                )
        return responses

    def simGetCollisionInfo(self, vehicle_name=""):
        return SimpleNamespace(
            has_collided=self.collided,
            object_name="wall",
            time_stamp=self.collision_timestamp if self.collided else 0,
        )


class FakeSDK:
    ImageType = SimpleNamespace(Scene=0, DepthPlanar=1)
    ImageRequest = FakeImageRequest
    Pose = FakePose
    Vector3r = FakeVector3r

    def __init__(self):
        self.client = FakeClient()
        self.MultirotorClient = lambda *args, **kwargs: self.client
        self.VehicleClient = lambda *args, **kwargs: self.client

    @staticmethod
    def to_quaternion(pitch, roll, yaw):
        return (pitch, roll, yaw)


class AirSimSimulatorTest(unittest.TestCase):
    def test_direct_reset_step_collision_and_close(self):
        sdk = FakeSDK()
        simulator = AirSimSimulator(
            env_name="scene",
            env_root="/unused",
            launch=False,
            sdk=sdk,
            settle_time_s=0,
            configure_camera_poses=True,
        )
        simulator._decode_image = lambda response: "rgb"
        episode = EpisodeSpec("ep", "scene", "go", Pose(1, 2, 3, 0), (5, 2, 3), 4)
        observation = simulator.reset(episode)
        native = sdk.client.poses[-1][0].position
        self.assertEqual((native.x_val, native.y_val, native.z_val), (1, -2, -3))
        self.assertEqual(observation.rgb, "rgb")
        self.assertEqual(sdk.client.camera_poses[0][0], "front_custom")

        sdk.client.collide_on_next_pose = True
        transition = simulator.execute(CanonicalAction(2, 0, 0))
        self.assertEqual(transition.observation.pose.xyz(), (3, 2, 3))
        self.assertTrue(transition.collision)
        self.assertFalse(sdk.client.poses[-1][1])
        simulator.close()
        self.assertEqual(sdk.client.armed, [True, False])
        self.assertEqual(sdk.client.api, [True, False])

    def test_stale_idle_collision_is_not_attributed_to_the_next_action(self):
        sdk = FakeSDK()
        simulator = AirSimSimulator(
            env_name="scene", env_root="/unused", launch=False, sdk=sdk, settle_time_s=0
        )
        simulator._decode_image = lambda response: "rgb"
        episode = EpisodeSpec("ep", "scene", "go", Pose(0, 0, 3, 0), (2, 0, 3), 2)
        simulator.reset(episode)
        sdk.client.collided = True
        sdk.client.collision_timestamp = 9
        transition = simulator.execute(CanonicalAction(1, 0, 0))
        self.assertFalse(transition.collision)
        self.assertTrue(transition.info["collision"]["ignored_stale_collision"])
        simulator.close()

    def test_gs_preset_uses_gs_coordinate_convention(self):
        sdk = FakeSDK()
        simulator = GSAirSimSimulator(
            env_name="scene",
            env_root="/unused",
            launch=False,
            sdk=sdk,
            settle_time_s=0,
            position_scale=0.2,
        )
        simulator._decode_image = lambda response: "mirrored-rgb"
        episode = EpisodeSpec("ep", "scene", "go", Pose(1, 2, 3, 0.5), (1, 2, 3), 0)
        simulator.reset(episode)
        native = sdk.client.poses[-1][0]
        self.assertEqual(
            (native.position.x_val, native.position.y_val, native.position.z_val),
            (0.2, 0.4, -0.6000000000000001),
        )
        self.assertEqual(native.orientation[2], 0.5)
        self.assertEqual(simulator._observation().rgb, "mirrored-rgb")

    def test_depth_capture_is_exposed_without_polluting_http_auxiliary_state(self):
        sdk = FakeSDK()
        simulator = AirSimSimulator(
            env_name="scene",
            env_root="/unused",
            launch=False,
            sdk=sdk,
            settle_time_s=0,
            capture_depth=True,
        )
        episode = EpisodeSpec("ep", "scene", "go", Pose(0, 0, 3, 0), (1, 0, 3), 1)
        observation = simulator.reset(episode)
        self.assertEqual(observation.info["depth"]["front"].shape, (1, 1))
        self.assertAlmostEqual(float(observation.info["depth"]["front"][0, 0]), 12.5)
        self.assertNotIn("depth", observation.auxiliary_state)
        simulator.close()
