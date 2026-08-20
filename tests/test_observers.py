import unittest

from aeroloop.simulators.mock import MockSimulator
from aeroloop.observers import RolloutObserver
from aeroloop.policies.mock import MockPolicy
from aeroloop.runner import RolloutConfig, RolloutRunner
from aeroloop.types import EpisodeSpec, Pose, TerminationReason


class SpyObserver(RolloutObserver):
    def __init__(self, abort_after=None):
        self.abort_after = abort_after
        self.starts = 0
        self.steps = 0
        self.ends = 0

    def on_episode_start(self, episode, observation):
        self.starts += 1

    def on_step(self, episode, observation, record):
        self.steps += 1
        return self.abort_after is None or self.steps < self.abort_after

    def on_episode_end(self, episode, observation, result):
        self.ends += 1

    def artifacts(self):
        return {"trace": "observer"}


class ObserverTest(unittest.TestCase):
    def test_lifecycle_and_artifacts(self):
        observer = SpyObserver()
        episode = EpisodeSpec("ep", "mock", "go", Pose(0, 0, 0, 0), (2, 0, 0), 2)
        result = RolloutRunner(
            MockSimulator(),
            MockPolicy(action=(1, 0, 0, 0, 0)),
            RolloutConfig(max_steps=2),
            observers=[observer],
        ).run_episode(episode)
        self.assertEqual((observer.starts, observer.steps, observer.ends), (1, 2, 1))
        self.assertEqual(result.artifacts, {"trace": "observer"})

    def test_observer_can_abort_episode(self):
        observer = SpyObserver(abort_after=1)
        episode = EpisodeSpec("ep", "mock", "go", Pose(0, 0, 0, 0), (10, 0, 0), 10)
        result = RolloutRunner(
            MockSimulator(), MockPolicy(), RolloutConfig(max_steps=10), observers=[observer]
        ).run_episode(episode)
        self.assertEqual(result.termination_reason, TerminationReason.USER_ABORT)
        self.assertEqual(result.metrics["steps_taken"], 1)


if __name__ == "__main__":
    unittest.main()
