import json
import tempfile
import unittest
from importlib.metadata import EntryPoint, EntryPoints
from pathlib import Path
from unittest.mock import patch

from uav_eval.cli import _load_episodes, run
from uav_eval.envs.mock import MockEnvironment
from uav_eval.factories import build_custom_metrics, build_environment, build_policy
from uav_eval.metrics import Metric
from uav_eval.policies.mock import MockPolicy
from uav_eval.types import EpisodeSpec, Pose


class CustomEnvironment(MockEnvironment):
    def __init__(self, env_name, **kwargs):
        super().__init__(**kwargs)
        self.requested_env_name = env_name


class CustomMetric(Metric):
    name = "fixture"

    def reset(self, episode):
        pass

    def finalize(self, termination, final_pose, step_count):
        return {"value": 1}


def load_fixture_episodes(prefix="fixture"):
    return [EpisodeSpec(f"{prefix}-ep", "mock", "go", Pose(0, 0, 0, 0), (1, 0, 0), 1)]


class ExtensionTest(unittest.TestCase):
    def test_environment_can_be_loaded_from_import_path(self):
        environment = build_environment({"type": "custom", "entrypoint": "test_extensions:CustomEnvironment"}, "mock")
        self.assertIsInstance(environment, CustomEnvironment)
        self.assertEqual(environment.requested_env_name, "mock")

    def test_policy_can_be_loaded_from_import_path(self):
        policy = build_policy(
            {"type": "custom", "entrypoint": "uav_eval.policies.mock:MockPolicy", "kwargs": {"chunk_size": 2}}
        )
        self.assertIsInstance(policy, MockPolicy)
        self.assertEqual(policy.chunk_size, 2)

    def test_policy_can_be_loaded_from_installed_entry_point(self):
        entry_points = EntryPoints(
            [
                EntryPoint(
                    name="installed-mock",
                    value="uav_eval.policies.mock:MockPolicy",
                    group="uav_eval.policies",
                )
            ]
        )
        with patch("uav_eval.extensions.metadata.entry_points", return_value=entry_points):
            policy = build_policy({"type": "installed-mock"})
        self.assertIsInstance(policy, MockPolicy)

    def test_metric_can_be_loaded_from_import_path(self):
        metrics = build_custom_metrics([{"entrypoint": "test_extensions:CustomMetric"}])
        self.assertEqual(len(metrics), 1)
        self.assertIsInstance(metrics[0], CustomMetric)

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
                    "environment": {"type": "mock"},
                    "policy": {"type": "mock"},
                    "rollout": {"max_steps": 1},
                    "metrics": {"custom": [{"entrypoint": "test_extensions:CustomMetric"}]},
                    "output": {"jsonl": str(output)},
                }
            )
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            episode_row = next(row for row in rows if row["type"] == "episode")
            self.assertEqual(episode_row["fixture/value"], 1)


if __name__ == "__main__":
    unittest.main()
