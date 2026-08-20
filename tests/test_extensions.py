import json
import tempfile
import unittest
from importlib.metadata import EntryPoint, EntryPoints
from pathlib import Path
from unittest.mock import patch

from aeroloop.cli import _load_episodes, run
from aeroloop.simulators.mock import MockSimulator
from aeroloop.simulators.airsim import AirSimSimulator, GSAirSimSimulator
from aeroloop.simulators.unrealcv import UnrealCVSimulator
from aeroloop.factories import build_custom_metrics, build_environment, build_observers, build_policy, build_simulator
from aeroloop.metrics import Metric
from aeroloop.observers import RolloutObserver
from aeroloop.policies.mock import MockPolicy
from aeroloop.protocols import EnvironmentAdapter, SimulatorAdapter
from aeroloop.types import EpisodeSpec, Pose


class CustomSimulator(MockSimulator):
    def __init__(self, env_name, **kwargs):
        super().__init__(**kwargs)
        self.requested_env_name = env_name


class CustomMetric(Metric):
    name = "fixture"

    def reset(self, episode):
        pass

    def finalize(self, termination, final_pose, step_count):
        return {"value": 1}


class CustomCollector(RolloutObserver):
    def __init__(self):
        self.steps = 0

    def on_step(self, episode, observation, record):
        self.steps += 1
        return True

    def artifacts(self):
        return {"collector_steps": self.steps}


def load_fixture_episodes(prefix="fixture"):
    return [EpisodeSpec(f"{prefix}-ep", "mock", "go", Pose(0, 0, 0, 0), (1, 0, 0), 1)]


class ExtensionTest(unittest.TestCase):
    def test_environment_adapter_is_a_compatibility_alias(self):
        self.assertIs(EnvironmentAdapter, SimulatorAdapter)

    def test_environment_can_be_loaded_from_import_path(self):
        environment = build_environment({"type": "custom", "entrypoint": "test_extensions:CustomSimulator"}, "mock")
        self.assertIsInstance(environment, CustomSimulator)
        self.assertEqual(environment.requested_env_name, "mock")

    def test_simulator_can_be_loaded_from_import_path(self):
        simulator = build_simulator({"type": "custom", "entrypoint": "test_extensions:CustomSimulator"}, "mock")
        self.assertIsInstance(simulator, CustomSimulator)

    def test_builtin_direct_simulator_types(self):
        common = {"kwargs": {"env_root": "/unused", "launch": False}}
        airsim = build_simulator({"type": "airsim", **common}, "scene")
        gs = build_simulator({"type": "gs_airsim", **common}, "scene")
        unrealcv = build_simulator({"type": "unrealcv", **common}, "scene")
        self.assertIsInstance(airsim, AirSimSimulator)
        self.assertIsInstance(gs, GSAirSimSimulator)
        self.assertIsInstance(unrealcv, UnrealCVSimulator)

    def test_simulator_can_be_loaded_from_installed_entry_point(self):
        entry_points = EntryPoints(
            [
                EntryPoint(
                    name="installed-simulator",
                    value="test_extensions:CustomSimulator",
                    group="aeroloop.simulators",
                )
            ]
        )
        with patch("aeroloop.extensions.metadata.entry_points", return_value=entry_points):
            simulator = build_simulator({"type": "installed-simulator"}, "mock")
        self.assertIsInstance(simulator, CustomSimulator)

    def test_policy_can_be_loaded_from_import_path(self):
        policy = build_policy(
            {"type": "custom", "entrypoint": "aeroloop.policies.mock:MockPolicy", "kwargs": {"chunk_size": 2}}
        )
        self.assertIsInstance(policy, MockPolicy)
        self.assertEqual(policy.chunk_size, 2)

    def test_policy_can_be_loaded_from_installed_entry_point(self):
        entry_points = EntryPoints(
            [
                EntryPoint(
                    name="installed-mock",
                    value="aeroloop.policies.mock:MockPolicy",
                    group="aeroloop.policies",
                )
            ]
        )
        with patch("aeroloop.extensions.metadata.entry_points", return_value=entry_points):
            policy = build_policy({"type": "installed-mock"})
        self.assertIsInstance(policy, MockPolicy)

    def test_metric_can_be_loaded_from_import_path(self):
        metrics = build_custom_metrics([{"entrypoint": "test_extensions:CustomMetric"}])
        self.assertEqual(len(metrics), 1)
        self.assertIsInstance(metrics[0], CustomMetric)

    def test_observer_can_be_loaded_from_import_path(self):
        observers = build_observers([{"entrypoint": "test_extensions:CustomCollector"}])
        self.assertEqual(len(observers), 1)
        self.assertIsInstance(observers[0], CustomCollector)

    def test_episode_source_can_be_loaded_from_import_path(self):
        episodes = _load_episodes(
            {
                "benchmark": {
                    "source": "custom",
                    "entrypoint": "test_extensions:load_fixture_episodes",
                    "kwargs": {"prefix": "external"},
                }
            }
        )
        self.assertEqual(episodes[0].episode_id, "external-ep")

    def test_custom_metric_runs_through_cli_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "result.jsonl"
            run(
                {
                    "benchmark": {
                        "source": "inline",
                        "episodes": [
                            {
                                "episode_id": "ep",
                                "env_name": "mock",
                                "target_position": [1, 0, 0],
                            }
                        ],
                    },
                    "simulator": {"type": "mock"},
                    "policy": {"type": "mock"},
                    "rollout": {"max_steps": 1},
                    "metrics": {"custom": [{"entrypoint": "test_extensions:CustomMetric"}]},
                    "observers": [{"entrypoint": "test_extensions:CustomCollector"}],
                    "output": {"jsonl": str(output)},
                }
            )
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            episode_row = next(row for row in rows if row["type"] == "episode")
            self.assertEqual(episode_row["fixture/value"], 1)
            self.assertEqual(episode_row["artifacts"]["collector_steps"], 1)


if __name__ == "__main__":
    unittest.main()
