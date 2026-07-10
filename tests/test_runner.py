import unittest

from uav_eval.envs.mock import MockEnvironment
from uav_eval.metrics import MetricConfig
from uav_eval.policies.mock import MockPolicy
from uav_eval.runner import RolloutConfig, RolloutRunner
from uav_eval.types import EpisodeSpec, Pose, TerminationReason


def episode(target_x=2.0):
    return EpisodeSpec(
        episode_id="ep",
        env_name="mock",
        instruction="fly forward",
        start_pose=Pose(0, 0, 0, 0),
        target_position=(target_x, 0, 0),
        reference_path_length=target_x,
    )


class RunnerTest(unittest.TestCase):
    def test_replan_every_step_and_explicit_stop(self):
        policy = MockPolicy(action=(1, 0, 0, 0, 0), chunk_size=4, stop_after=2)
        runner = RolloutRunner(
            MockEnvironment(),
            policy,
            RolloutConfig(max_steps=10, execution_horizon=1),
            MetricConfig(success_distance=0.1, distance_mode="endpoint_3d", require_stop_for_success=True),
        )
        result = runner.run_episode(episode())
        self.assertEqual(result.termination_reason, TerminationReason.STOP)
        self.assertEqual(result.metrics["success"], 1)
        self.assertEqual(result.metrics["inference_calls"], 3)
        self.assertEqual(result.metrics["steps_taken"], 3)
        self.assertAlmostEqual(result.metrics["path_length"], 2)

    def test_native_chunk_only_calls_policy_once(self):
        policy = MockPolicy(action=(1, 0, 0, 0, 0), chunk_size=4)
        result = RolloutRunner(
            MockEnvironment(), policy, RolloutConfig(max_steps=4, execution_horizon=None)
        ).run_episode(episode(target_x=4))
        self.assertEqual(result.metrics["inference_calls"], 1)
        self.assertEqual(result.termination_reason, TerminationReason.MAX_STEPS)

    def test_collision_terminates(self):
        result = RolloutRunner(
            MockEnvironment(collision_x=2),
            MockPolicy(action=(1, 0, 0, 0, 0)),
            RolloutConfig(max_steps=10, terminate_on_collision=True),
        ).run_episode(episode(target_x=10))
        self.assertEqual(result.termination_reason, TerminationReason.COLLISION)
        self.assertEqual(result.metrics["steps_taken"], 2)
        self.assertTrue(result.metrics["collision"])


if __name__ == "__main__":
    unittest.main()
