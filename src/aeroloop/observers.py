from __future__ import annotations

from typing import Any, Mapping

from .types import EpisodeResult, EpisodeSpec, Observation, StepRecord


class RolloutObserver:
    """Optional rollout side effect such as UI, video, or telemetry.

    Returning ``False`` from ``on_step`` requests a clean user-abort of the
    current episode. Observer failures are non-fatal by default.
    """

    def on_episode_start(self, episode: EpisodeSpec, observation: Observation) -> None:
        pass

    def on_step(self, episode: EpisodeSpec, observation: Observation, record: StepRecord) -> bool:
        return True

    def on_episode_end(self, episode: EpisodeSpec, observation: Observation, result: EpisodeResult) -> None:
        pass

    def artifacts(self) -> Mapping[str, Any]:
        return {}

    def close(self) -> None:
        pass
