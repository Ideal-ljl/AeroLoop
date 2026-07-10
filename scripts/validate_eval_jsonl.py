#!/usr/bin/env python3
"""Fail a DLC evaluation when its JSONL contains incomplete/error episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()

    episodes = []
    with args.jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("type") == "episode":
                episodes.append(row)

    errors = [row for row in episodes if row.get("termination_reason") == "error" or row.get("error")]
    if len(episodes) != args.expected or errors:
        examples = [f"{row.get('episode_id')}: {row.get('error')}" for row in errors[:3]]
        raise SystemExit(
            f"invalid evaluation: episodes={len(episodes)}/{args.expected}, "
            f"errors={len(errors)}; examples={examples}"
        )
    print(f"validated {len(episodes)} completed episodes: {args.jsonl}")


if __name__ == "__main__":
    main()
