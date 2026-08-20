import unittest

from aeroloop.policies.l1_oracle import L1OraclePolicy, l1_oracle_guidance
from aeroloop.protocols import PolicyAdapter
from aeroloop.types import ActionChunk, CanonicalAction, EpisodeSpec, Observation, PolicyInput, Pose


class CapturingPolicy(PolicyAdapter):
    def reset(self, episode):
        self.episode = episode

    def predict(self, policy_input):
        self.policy_input = policy_input
        return ActionChunk((CanonicalAction(1, 0, 0),), {"model": "base"})


class L1OraclePolicyTest(unittest.TestCase):
    def test_stage_rules_use_canonical_heading_and_height(self):
        cases = [
            ([(0, 0, 0), (20, 0, 0)], Pose(0, 0, 0, 0), 0, "take off"),
            ([(0, 0, 0), (20, 0, 0)], Pose(0, 0, 0, 0), 6, "cruise"),
            ([(0, 0, 0), (0, 20, 0)], Pose(0, 0, 0, 0), 6, "left"),
            ([(0, 0, 0), (0, -20, 0)], Pose(0, 0, 0, 0), 6, "right"),
            ([(0, 0, 0), (0, 0, 20)], Pose(0, 0, 0, 0), 6, "take off"),
            ([(0, 0, 10), (20, 0, 0)], Pose(15, 0, 10, 0), 6, "landing"),
        ]
        for trajectory, pose, step, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(l1_oracle_guidance(trajectory, pose, step).stage, expected)

    def test_wrapper_injects_l1_context_and_records_oracle_metadata(self):
        base = CapturingPolicy()
        policy = L1OraclePolicy(base)
        episode = EpisodeSpec(
            "ep",
            "scene",
            "find target",
            Pose(0, 0, 0, 0),
            (0, 20, 0),
            20,
            metadata={"trajectory": [[0, 0, 0], [0, 20, 0]]},
        )
        policy.reset(episode)
        observation = Observation(None, Pose(0, 0, 0, 0), (0, 0, 0, 0), 6)

        chunk = policy.predict(PolicyInput(episode, observation, policy_context={"existing": True}))

        self.assertEqual(base.policy_input.policy_context["assistant_level"], "L1")
        self.assertEqual(base.policy_input.policy_context["assistant_stage"], "left")
        self.assertTrue(base.policy_input.policy_context["existing"])
        self.assertEqual(chunk.metadata["assistant_stage"], "left")
        self.assertEqual(chunk.metadata["assistant_reference_point"], [0.0, 20.0, 0.0])

    def test_missing_reference_trajectory_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "reference trajectory"):
            l1_oracle_guidance([], Pose(0, 0, 0, 0), 0)


if __name__ == "__main__":
    unittest.main()
