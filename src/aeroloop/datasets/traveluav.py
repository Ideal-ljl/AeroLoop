from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ..types import EpisodeSpec, Pose
from .common import load_json, path_length, rewrite_path, yaw_from_quaternion


def load_traveluav_episodes(
    path: str | Mapping[str, str] | Sequence[str],
    *,
    dataset_root: str,
    path_rewrites: Mapping[str, str] | None = None,
    include_envs: str | Sequence[str] | None = None,
    deduplicate: bool = True,
    limit: int | None = None,
) -> list[EpisodeSpec]:
    """Load TravelUAV split rows and their referenced ``merged_data.json`` files.

    ``path`` may be one manifest, a list of manifests, or a mapping such as
    ``{"seen": "seen_valset.json", "unseen": "unseen_valset.json"}``.  The
    mapping form preserves the official Seen/Unseen split while still
    deduplicating the many frame rows into one closed-loop episode.
    """

    if isinstance(path, Mapping):
        sources = [(str(label), str(source)) for label, source in path.items()]
    elif isinstance(path, (str, Path)):
        source = str(path)
        label = "unseen" if "unseen" in Path(source).stem.lower() else "seen"
        sources = [(label, source)]
    else:
        sources = []
        for source in path:
            source = str(source)
            label = "unseen" if "unseen" in Path(source).stem.lower() else "seen"
            sources.append((label, source))

    if isinstance(include_envs, str):
        included = {include_envs}
    elif include_envs is None:
        included = None
    else:
        included = {str(name) for name in include_envs}

    selected: list[tuple[str, int | None, str]] = []
    seen = set()
    for split_label, source in sources:
        split_rows = load_json(source)
        for row in split_rows:
            relative = str(row.get("json") or row.get("path") or "")
            if not relative:
                raise ValueError("TravelUAV split row requires 'json'")
            parts = Path(relative).parts
            if len(parts) < 3:
                raise ValueError(f"TravelUAV path must end in <map>/<sequence>/merged_data.json: {relative}")
            if included is not None and parts[-3] not in included:
                continue
            key = relative if deduplicate else (relative, row.get("frame"))
            if key in seen:
                continue
            seen.add(key)
            selected.append((relative, row.get("frame"), split_label))
            if limit is not None and len(selected) >= limit:
                break
        if limit is not None and len(selected) >= limit:
            break

    episodes = []
    for relative, frame, split_label in selected:
        merged_path = rewrite_path(relative, dataset_root, path_rewrites)
        if not merged_path.exists():
            raise FileNotFoundError(f"TravelUAV merged trajectory not found: {merged_path}")
        merged = load_json(merged_path)
        states = merged.get("trajectory_raw_detailed") or merged.get("trajectory_raw") or []
        if not states:
            raise ValueError(f"TravelUAV trajectory is empty: {merged_path}")
        positions = [state["position"] for state in states]
        start_index = 0 if deduplicate or frame is None else min(max(int(frame) - 1, 0), len(states) - 1)
        start = states[start_index]
        instruction = str(((merged.get("conversations") or [{}])[0]).get("value", ""))
        instruction = instruction.removeprefix("<image>\n")
        scene = merged_path.parent.parent.name
        sequence = merged_path.parent.name
        reference_length = path_length(positions[start_index:])
        normalized_label = split_label.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized_label == "seen":
            benchmark_split = "seen"
        elif scene in {"Carla_Town03", "ModularPark"}:
            benchmark_split = "unseen_map"
        else:
            benchmark_split = "unseen_object"
        metadata: dict[str, Any] = {
            "dataset": "traveluav",
            "merged_path": str(merged_path),
            "trajectory_dir": str(merged_path.parent),
            "trajectory": [list(map(float, point[:3])) for point in positions],
            "split_frame": frame,
            "start_frame": start_index,
            "official_reference_path_offset": 20.0,
            "source_split": split_label,
            "benchmark_split": benchmark_split,
            "difficulty": "easy" if reference_length <= 250.0 else "hard",
        }
        episodes.append(
            EpisodeSpec(
                episode_id=f"traveluav:{scene}:{sequence}" + ("" if deduplicate else f":{start_index}"),
                env_name=scene,
                instruction=instruction,
                start_pose=Pose(
                    *map(float, start["position"][:3]), yaw_from_quaternion(start.get("orientation", (0, 0, 0, 1)))
                ),
                target_position=tuple(map(float, positions[-1][:3])),
                reference_path_length=reference_length,
                metadata=metadata,
            )
        )
    return episodes
