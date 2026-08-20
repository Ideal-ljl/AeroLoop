from __future__ import annotations

import sys
from dataclasses import dataclass

from ..protocols import PolicyAdapter
from ..types import ActionChunk, EpisodeSpec, PolicyInput


@dataclass
class CkptHarnessPolicy(PolicyAdapter):
    """Compatibility adapter for Ab_ex/ckpt/harness.

    The concrete harness wrappers still need model-specific repairs; this class
    keeps those repairs outside the benchmark runner.
    """

    model_name: str
    ckpt_dir: str
    workspace_root: str
    device: str = "cuda"
    dtype: str = "bfloat16"
    name: str = "ckpt_harness"

    def __post_init__(self) -> None:
        if self.workspace_root not in sys.path:
            sys.path.insert(0, self.workspace_root)
        from ckpt.harness import build_model

        self.model = build_model(self.model_name, self.ckpt_dir, device=self.device, dtype=self.dtype)

    def reset(self, episode: EpisodeSpec) -> None:
        self.model.reset(episode.instruction)

    def predict(self, policy_input: PolicyInput) -> ActionChunk:
        from ckpt.harness import Observation as HarnessObservation

        obs = policy_input.observation
        output = self.model.predict(
            HarnessObservation(frames=[obs.rgb], instruction=policy_input.episode.instruction, state=obs.relative_state)
        )
        return ActionChunk.from_rows(output.tolist(), {"source": "ckpt.harness", "model": self.model_name})
