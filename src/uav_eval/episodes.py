from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .geometry import distance
from .types import EpisodeSpec, Pose


def _env_name_from_path(path: Path) -> str:
    for part in reversed(path.parts):
        if part.startswith("env_"):
            return part
    if len(path.parts) >= 3:
        return path.parts[-3]
    raise ValueError(f"cannot infer environment name from {path}")


def _instruction(row: Mapping[str, Any]) -> str:
    if row.get("gpt_instruction"):
        return str(row["gpt_instruction"])
    caption = row.get("vla_caption")
    if isinstance(caption, Mapping) and caption.get("result"):
        return str(caption["result"])
    return str(row.get("instruction", ""))


def load_airbrain_episodes(
    eval_config: str | Path,
    *,
    repo_id: str | None = None,
    max_samples: int | None = None,
) -> list[EpisodeSpec]:
    rows = json.loads(Path(eval_config).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("AirBrain eval config must contain a JSON list")
    episodes: list[EpisodeSpec] = []
    per_env: dict[str, int] = {}
    for row in rows:
        data_path = Path(row["path"])
        env_name = _env_name_from_path(data_path)
        if repo_id and env_name != repo_id:
            continue
        used = per_env.get(env_name, 0)
        if max_samples is not None and used >= max_samples:
            continue
        pose_file = data_path / "pose_bbox_updated.json"
        pose_info = json.loads(pose_file.read_text(encoding="utf-8"))
        actions = pose_info["actions"]
        positions = [tuple(map(float, item["pos"][:3])) for item in actions]
        yaws = [float(item["yaw"]) for item in actions]
        start = positions[0]
        end = positions[-1]
        start_pose = Pose(*start, yaws[0])
        has_two_landmarks = "aim_landmark_1" in pose_info
        mid = tuple(map(float, pose_info["aim_landmark_0"]["position"][:3])) if has_two_landmarks else None
        building_key = "aim_landmark_1" if has_two_landmarks else "aim_landmark_0"
        building = tuple(map(float, pose_info[building_key]["position"][:3]))
        reference = distance(start, mid, 3) + distance(mid, end, 3) if mid is not None else distance(start, end, 3)
        bbox_info = actions[0].get("bbox_info") or []
        target_id = int(bbox_info[-1]["id"]) if bbox_info else None
        episode_id = f"{env_name}:{data_path.name}"
        episodes.append(
            EpisodeSpec(
                episode_id=episode_id,
                env_name=env_name,
                instruction=_instruction(row),
                start_pose=start_pose,
                target_position=end,
                reference_path_length=reference,
                building_position=building,
                target_id=target_id,
                metadata={
                    "path": str(data_path),
                    "pose_file": str(pose_file),
                    "pitch": -45.0 if "high" in str(data_path) else 0.0,
                    "target_boxes_info": bbox_info,
                },
            )
        )
        per_env[env_name] = used + 1
    return episodes


def load_inline_episodes(rows: Iterable[Mapping[str, Any]]) -> list[EpisodeSpec]:
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
                building_position=tuple(row["building_position"][:3]) if row.get("building_position") else None,
                target_id=row.get("target_id"),
                metadata=dict(row.get("metadata", {})),
            )
        )
    return episodes
