import unittest

from uav_eval.envs.mock import MockEnvironment
from uav_eval.metrics import Metric, MetricConfig, aggregate_results
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

    def test_custom_metric_is_namespaced(self):
        class MotionMetric(Metric):
            name = "motion"

            def reset(self, episode):
                self.total = 0.0

            def update(self, before, after, *, action, collision, info):
                self.total += abs(action.dx) + abs(action.dy) + abs(action.dz)

            def finalize(self, termination, final_pose, step_count):
                return {"effort": self.total, "steps": step_count}

        result = RolloutRunner(
            MockEnvironment(),
            MockPolicy(action=(1, 0, 0, 0, 0)),
            RolloutConfig(max_steps=2),
            custom_metrics=[MotionMetric()],
        ).run_episode(episode())
        self.assertEqual(result.metrics["motion/effort"], 2.0)
        self.assertEqual(result.metrics["motion/steps"], 2)

    def test_custom_metric_failure_marks_only_the_episode_as_error(self):
        class BrokenMetric(Metric):
            name = "broken"

            def reset(self, episode):
                pass

            def finalize(self, termination, final_pose, step_count):
                raise RuntimeError("broken fixture")

        result = RolloutRunner(
            MockEnvironment(), MockPolicy(), RolloutConfig(max_steps=1), custom_metrics=[BrokenMetric()]
        ).run_episode(episode())
        self.assertEqual(result.termination_reason, TerminationReason.ERROR)
        self.assertIn("metric finalize failed", result.error)
        self.assertIn("success", result.metrics)

    def test_numeric_custom_metrics_are_aggregated(self):
        summary = aggregate_results(
            [
                {"success": 1, "motion/effort": 2.0},
                {"success": 0, "motion/effort": 4.0},
            ]
        )
        self.assertEqual(summary["motion/effort"], 3.0)


if __name__ == "__main__":
    unittest.main()
