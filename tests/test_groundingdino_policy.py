import unittest

import numpy as np

from aeroloop.policies.groundingdino import GroundingDinoStopPolicy, qualifying_detections
from aeroloop.protocols import PolicyAdapter
from aeroloop.types import ActionChunk, CanonicalAction, EpisodeSpec, Observation, PolicyInput, Pose


class FakeBasePolicy(PolicyAdapter):
    def __init__(self, actions=None):
        self.calls = 0
        self.actions = tuple(actions or (CanonicalAction(1, 0, 0),))

    def reset(self, episode):
        self.episode = episode

    def predict(self, policy_input):
        self.calls += 1
        return ActionChunk(self.actions, {"model": "base"})


class FakeDetector:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def detect(self, image, prompt, depth):
        self.calls.append((image.shape, prompt, depth.shape))
        return list(self.rows)


class SequencedDetector(FakeDetector):
    def __init__(self, rows_by_call):
        super().__init__([])
        self.rows_by_call = list(rows_by_call)

    def detect(self, image, prompt, depth):
        self.calls.append((image.shape, prompt, depth.shape))
        return list(self.rows_by_call.pop(0)) if self.rows_by_call else []


class GroundingDinoPolicyTest(unittest.TestCase):
    def test_original_box_and_center_depth_rule(self):
        depth = np.full((100, 200), 30.0, dtype=np.float32)
        depth[50, 100] = 12.0
        accepted = qualifying_detections(
            [[80, 30, 120, 70], [0, 0, 190, 90]],
            [0.9, 0.99],
            ["tower", "oversized"],
            depth,
            image_width=200,
            image_height=100,
        )
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["label"], "tower")
        self.assertEqual(accepted[0]["center_depth"], 12.0)

    def test_detector_stops_after_first_executed_step(self):
        base = FakeBasePolicy()
        detector = FakeDetector([{"score": 0.9, "center_depth": 10.0}])
        policy = GroundingDinoStopPolicy(base, detector=detector, min_step=1)
        episode = EpisodeSpec(
            "ep",
            "scene",
            "find it",
            Pose(0, 0, 0, 0),
            (1, 0, 0),
            1,
            metadata={"grounding_prompt": "red tower"},
        )
        policy.reset(episode)
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        depth = np.ones((8, 8), dtype=np.float32)
        initial = Observation(image, Pose(0, 0, 0, 0), (0, 0, 0, 0), 0, info={"depth": {"front": depth}})
        self.assertEqual(policy.predict(PolicyInput(episode, initial)).actions[0].stop, 0.0)
        self.assertEqual(base.calls, 1)
        after_step = Observation(image, Pose(1, 0, 0, 0), (1, 0, 0, 0), 1, info={"depth": {"front": depth}})
        result = policy.predict(PolicyInput(episode, after_step))
        self.assertEqual(result.actions[0].stop, 1.0)
        self.assertTrue(result.metadata["grounding_stop"])
        self.assertEqual(base.calls, 1)
        self.assertEqual(detector.calls[0][1], "red tower")

    def test_reuses_base_chunk_while_detector_runs_every_step(self):
        base = FakeBasePolicy(
            (
                CanonicalAction(1, 0, 0),
                CanonicalAction(2, 0, 0),
                CanonicalAction(3, 0, 0),
            )
        )
        detector = SequencedDetector([[], [], []])
        policy = GroundingDinoStopPolicy(base, detector=detector, min_step=1)
        episode = EpisodeSpec(
            "ep",
            "scene",
            "find it",
            Pose(0, 0, 0, 0),
            (10, 0, 0),
            10,
            metadata={"grounding_prompt": "red tower"},
        )
        policy.reset(episode)
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        depth = np.ones((8, 8), dtype=np.float32)

        actions = []
        metadata = []
        for step in range(4):
            observation = Observation(
                image,
                Pose(step, 0, 0, 0),
                (step, 0, 0, 0),
                step,
                info={"depth": {"front": depth}},
            )
            result = policy.predict(PolicyInput(episode, observation))
            actions.append(result.actions[0].dx)
            metadata.append(result.metadata)

        self.assertEqual(actions, [1.0, 2.0, 3.0, 1.0])
        self.assertEqual(base.calls, 2)
        self.assertEqual(len(detector.calls), 3)
        self.assertFalse(metadata[0]["cached_base_action"])
        self.assertTrue(metadata[1]["cached_base_action"])
        self.assertEqual(metadata[2]["cached_actions_remaining"], 0)

    def test_detection_discards_remaining_cached_actions(self):
        base = FakeBasePolicy((CanonicalAction(1, 0, 0), CanonicalAction(2, 0, 0)))
        detector = SequencedDetector([[{"score": 0.9, "center_depth": 10.0}]])
        policy = GroundingDinoStopPolicy(base, detector=detector, min_step=1)
        episode = EpisodeSpec(
            "ep",
            "scene",
            "find it",
            Pose(0, 0, 0, 0),
            (10, 0, 0),
            10,
            metadata={"grounding_prompt": "red tower"},
        )
        policy.reset(episode)
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        depth = np.ones((8, 8), dtype=np.float32)
        initial = Observation(image, Pose(0, 0, 0, 0), (0, 0, 0, 0), 0, info={"depth": {"front": depth}})
        after_step = Observation(image, Pose(1, 0, 0, 0), (1, 0, 0, 0), 1, info={"depth": {"front": depth}})

        self.assertEqual(policy.predict(PolicyInput(episode, initial)).actions[0].dx, 1.0)
        stopped = policy.predict(PolicyInput(episode, after_step))

        self.assertEqual(stopped.actions[0].stop, 1.0)
        self.assertTrue(stopped.metadata["grounding_stop"])
        self.assertEqual(base.calls, 1)


if __name__ == "__main__":
    unittest.main()
