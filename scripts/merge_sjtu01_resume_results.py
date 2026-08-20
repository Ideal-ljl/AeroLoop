#!/usr/bin/env python3
"""Merge an interrupted SJTU01 evaluation with its exact resume suffix."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def episode_id(sample: dict[str, Any]) -> str:
    path = Path(sample["path"])
    return f"{path.parts[-3]}:{path.name}"


def mean(episodes: list[dict[str, Any]], key: str) -> float:
    return sum(float(episode[key]) for episode in episodes) / len(episodes)


def summary(episodes: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    return {
        "type": "overall_summary",
        "total_samples": len(episodes),
        "success": mean(episodes, "success"),
        "spl": mean(episodes, "spl"),
        "osr": mean(episodes, "osr"),
        "collision": mean(episodes, "collision"),
        "stop_called": mean(episodes, "stop_called"),
        "stop_success": mean(episodes, "stop_success"),
        "premature_stop": mean(episodes, "premature_stop"),
        "distance_to_goal": mean(episodes, "distance_to_goal"),
        "total_steps": sum(int(episode["steps_taken"]) for episode in episodes),
        "total_inference_calls": sum(int(episode["inference_calls"]) for episode in episodes),
        "output_file": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--suffix", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env", default="env_gs_sjtu01")
    args = parser.parse_args()

    prefix_rows = read_jsonl(args.prefix)
    suffix_rows = read_jsonl(args.suffix)
    prefix = [row for row in prefix_rows if row.get("type") == "episode"]
    suffix = [row for row in suffix_rows if row.get("type") == "episode"]
    episodes = prefix + suffix

    with args.dataset.open("r", encoding="utf-8") as handle:
        samples = json.load(handle)
    expected = [episode_id(sample) for sample in samples if args.env in Path(sample["path"]).parts]

    actual = [episode["episode_id"] for episode in episodes]
    if len(prefix) != 88 or len(suffix) != 12:
        raise SystemExit(f"expected 88+12 episodes, got {len(prefix)}+{len(suffix)}")
    if len(actual) != len(set(actual)):
        raise SystemExit("duplicate episode IDs in merged result")
    if actual != expected:
        for index, (got, want) in enumerate(zip(actual, expected)):
            if got != want:
                raise SystemExit(f"episode order mismatch at {index}: got {got}, expected {want}")
        raise SystemExit(f"episode count mismatch: got {len(actual)}, expected {len(expected)}")

    run_config = copy.deepcopy(prefix_rows[0])
    if run_config.get("type") != "run_config":
        raise SystemExit("prefix result does not begin with run_config")
    run_config["config"]["output"]["jsonl"] = str(args.output)
    run_config["merge_provenance"] = {
        "prefix": str(args.prefix),
        "suffix": str(args.suffix),
        "prefix_episodes": len(prefix),
        "suffix_episodes": len(suffix),
    }

    overall = summary(episodes, args.output)
    environment = {**overall, "type": "environment_summary", "env_name": args.env}
    environment.pop("output_file")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in [run_config, *episodes, environment, overall]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"merged {len(prefix)}+{len(suffix)}={len(episodes)} episodes into {args.output}")


if __name__ == "__main__":
    main()
