from __future__ import annotations

from typing import Iterable, Mapping

from .geometry import distance
from .types import EpisodeSpec, Pose


def load_inline_episodes(rows: Iterable[Mapping]) -> list[EpisodeSpec]:
    episodes = []
    for index, row in enumerate(rows):
        start = Pose.from_sequence(row.get("start_pose", [0, 0, 0, 0]))
        target = tuple(map(float, row.get("target_position", [10, 0, 0])[:3]))
        episodes.append(
            EpisodeSpec(
                episode_id=str(row.get("episode_id", f"mock-{index}")),
                env_name=str(row.get("env_name", "mock")),
                instruction=str(row.get("instruction", "fly to the target")),
                start_pose=start,
                target_position=target,
                reference_path_length=float(row.get("reference_path_length", distance(start.xyz(), target, 3))),
                metadata=dict(row.get("metadata", {})),
            )
        )
    return episodes
