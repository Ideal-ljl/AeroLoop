from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ..types import EpisodeSpec, Pose
from .common import load_json, path_length, rewrite_path


def _instruction(row: Mapping[str, Any]) -> str:
    if row.get("instruction"):
        return str(row["instruction"])
    if row.get("gpt_instruction"):
        return str(row["gpt_instruction"])
    caption = row.get("vla_caption") or {}
    if isinstance(caption, Mapping) and caption.get("result"):
        return str(caption["result"])
    return ""


def _yaw(action: Mapping[str, Any]) -> float:
    return float(action.get("yaw", action.get("orientation", [0, 0, 0, 0])[-1] if action.get("orientation") else 0))


def _grounding_prompt(row: Mapping[str, Any]) -> str:
    for item in reversed(row.get("vln_data") or []):
        if not isinstance(item, Mapping):
            continue
        for key in ("target_caption", "target_caption_env"):
            if item.get(key):
                return str(item[key]).strip()
    return _instruction(row).strip()


def load_openfly_episodes(
    path: str,
    *,
    dataset_root: str | None = None,
    path_rewrites: Mapping[str, str] | None = None,
    pose_filename: str = "pose_bbox_updated.json",
    env_name: str | None = None,
    include_envs: str | Sequence[str] | None = None,
    limit: int | None = None,
) -> list[EpisodeSpec]:
    """Load OpenFly's test-list plus per-trajectory pose metadata."""

    rows = load_json(path)
    if isinstance(include_envs, str):
        selected_envs = {include_envs}
    elif include_envs is None:
        selected_envs = None
    else:
        selected_envs = {str(value) for value in include_envs}
    episodes = []
    for index, row in enumerate(rows):
        trajectory_dir = rewrite_path(row["path"], dataset_root, path_rewrites)
        source_scene = next(
            (part for part in trajectory_dir.parts if part.startswith("env_")), trajectory_dir.parent.name
        )
        if selected_envs is not None and source_scene not in selected_envs:
            continue
        if limit is not None and len(episodes) >= limit:
            break
        pose_path = trajectory_dir / pose_filename
        if not pose_path.exists():
            raise FileNotFoundError(
                f"OpenFly trajectory metadata not found: {pose_path}; use dataset_root/path_rewrites for relocated data"
            )
        pose_info = load_json(pose_path)
        actions = pose_info.get("actions") or []
        points = [action["pos"] for action in actions if action.get("pos") is not None]
        if not points:
            raise ValueError(f"OpenFly trajectory has no actions[].pos: {pose_path}")
        target = points[-1]
        landmark = pose_info.get("aim_landmark_1") or pose_info.get("aim_landmark_0") or {}
        scene = env_name or source_scene
        metadata = {
            "dataset": "openfly",
            "source_row": index,
            "trajectory_dir": str(trajectory_dir),
            "pose_path": str(pose_path),
            "trajectory": [list(map(float, point[:3])) for point in points],
            "target_landmark": landmark,
            "grounding_prompt": _grounding_prompt(row),
            "raw": dict(row),
        }
        episodes.append(
            EpisodeSpec(
                episode_id=f"openfly:{scene}:{trajectory_dir.name}",
                env_name=scene,
                instruction=_instruction(row),
                start_pose=Pose(*map(float, points[0][:3]), _yaw(actions[0])),
                target_position=tuple(map(float, target[:3])),
                reference_path_length=path_length(points),
                metadata=metadata,
            )
        )
    return episodes
