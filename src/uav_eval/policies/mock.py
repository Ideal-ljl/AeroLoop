from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ..protocols import PolicyAdapter
from ..types import ActionChunk, EpisodeSpec, PolicyInput


@dataclass
class MockPolicy(PolicyAdapter):
    """Deterministic policy used to validate the platform without ML dependencies."""

    action: Sequence[float] = (1.0, 0.0, 0.0, 0.0, 0.0)
    chunk_size: int = 1
    stop_after: int | None = None
    name: str = "mock"
    calls: int = field(default=0, init=False)

    def reset(self, episode: EpisodeSpec) -> None:
        self.calls = 0

    def predict(self, policy_input: PolicyInput) -> ActionChunk:
        self.calls += 1
        rows = [list(self.action) for _ in range(self.chunk_size)]
        if self.stop_after is not None and len(policy_input.action_history) >= self.stop_after:
            rows[0] = [0.0, 0.0, 0.0, 0.0, 1.0]
        return ActionChunk.from_rows(rows, {"policy": self.name, "call": self.calls})
