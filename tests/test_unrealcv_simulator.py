import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from aeroloop.simulators.unrealcv import UnrealCVSimulator
from aeroloop.types import CanonicalAction, EpisodeSpec, Pose


class FakeUnrealCVClient:
    def __init__(self):
        self.commands = []
        self.disconnected = False

    def request(self, command):
        self.commands.append(command)
        if command == "vget /collision":
            return "true"
        return "ok"

    def disconnect(self):
        self.disconnected = True


class UnrealCVSimulatorTest(unittest.TestCase):
    def test_start_uses_unrealcv_connection_without_raw_port_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp) / "scene"
            (scene / "City_UE52/Binaries/Linux").mkdir(parents=True)
            (scene / "CitySample.sh").write_text("#!/bin/sh\n", encoding="utf-8")
            (scene / "City_UE52/Binaries/Linux/unrealcv.ini").write_text(
                "[UnrealCV.Core]\nPort=9000\n", encoding="utf-8"
            )
            client = FakeUnrealCVClient()
            client.connect = lambda timeout=1: timeout == 2
            process = SimpleNamespace(poll=lambda: None, returncode=None)
            simulator = UnrealCVSimulator(
                "scene",
                tmp,
                spawn_cameras=False,
                startup_grace_s=0,
            )
            with (
                patch("aeroloop.simulators.unrealcv.launch_process", return_value=process),
                patch(
                    "aeroloop.simulators.unrealcv.importlib.import_module",
                    return_value=SimpleNamespace(Client=lambda address: client),
                ),
            ):
                simulator._start()
            self.assertIs(simulator.client, client)

    def test_pose_conversion_collision_query_and_close(self):
        client = FakeUnrealCVClient()
        simulator = UnrealCVSimulator(
            env_name="scene",
            env_root="/unused",
            launch=False,
            client=client,
            settle_time_s=0,
            collision_query="vget /collision",
        )
        simulator._capture = lambda: {"front": "rgb"}
        episode = EpisodeSpec("ep", "scene", "go", Pose(1, 2, 3, 0), (2, 2, 3), 1)
        observation = simulator.reset(episode)
        self.assertEqual(observation.rgb, "rgb")
        self.assertIn("vset /camera/1/location 100.0 -200.0 300.0", client.commands)

        transition = simulator.execute(CanonicalAction(1, 0, 0))
        self.assertTrue(transition.collision)
        self.assertEqual(transition.observation.pose.xyz(), (2, 2, 3))
        simulator.close()
        self.assertTrue(client.disconnected)

    def test_ini_port_is_updated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "unrealcv.ini"
            path.write_text("Width=896\nPort=7777\n", encoding="utf-8")
            simulator = UnrealCVSimulator("scene", "/unused", launch=False, port=9000)
            simulator._set_ini_port(path)
            self.assertEqual(path.read_text(encoding="utf-8"), "Width=896\nPort=9000\n")


if __name__ == "__main__":
    unittest.main()
