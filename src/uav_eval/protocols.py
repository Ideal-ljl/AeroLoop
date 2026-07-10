from __future__ import annotations

from abc import ABC, abstractmethod

from .types import ActionChunk, CanonicalAction, EpisodeSpec, Observation, PolicyInput, Transition


class PolicyAdapter(ABC):
    name = "policy"

    @abstractmethod
    def reset(self, episode: EpisodeSpec) -> None:
        pass

    @abstractmethod
    def predict(self, policy_input: PolicyInput) -> ActionChunk:
        pass

    def on_action_executed(self, action: CanonicalAction, transition: Transition) -> None:
        pass

    def close(self) -> None:
        pass


class EnvironmentAdapter(ABC):
    name = "environment"

    @abstractmethod
    def reset(self, episode: EpisodeSpec) -> Observation:
        pass

    @abstractmethod
    def execute(self, action: CanonicalAction) -> Transition:
        pass

    def close(self) -> None:
        pass
