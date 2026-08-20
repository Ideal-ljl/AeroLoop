#!/usr/bin/env python3
"""Fail a DLC evaluation when its JSONL contains incomplete/error episodes."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _is_explicit_aerialvla_zero_action(step: dict) -> bool:
    """Return true when AerialVLA explicitly emitted its valid zero command.

    ``0 49 49`` dequantizes to forward=0, down=0 and yaw=0 for the
    repository's 99-bin codec.  A policy can genuinely get stuck emitting
    that command; it is a model failure that must remain in the benchmark,
    not an inference/infrastructure failure.  Requiring the strict raw text
    prevents malformed-output fallbacks from being accepted as legitimate.
    """
    metadata = step.get("policy_metadata") or {}
    if metadata.get("model") != "aerialvla":
        return False
    raw_output = metadata.get("raw_output")
    if not isinstance(raw_output, str):
        return False
    action_text = raw_output.rsplit("Action:", 1)[-1].split("</s>", 1)[0].strip()
    if re.fullmatch(r"0\s+49\s+49", action_text) is None:
        return False
    native = metadata.get("native_action") or []
    return len(native) >= 4 and all(abs(float(value)) <= 1e-8 for value in native[:3]) and not bool(native[3])


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
    degenerate = []
    for row in episodes:
        steps = row.get("steps") or []
        if steps and not row.get("stop_called"):
            actions = [step.get("action") or [] for step in steps]
            all_zero = actions and all(
                not any(abs(float(value)) > 1e-8 for value in action[:4]) for action in actions
            )
            explicit_policy_zero = all(_is_explicit_aerialvla_zero_action(step) for step in steps)
            if all_zero and not explicit_policy_zero:
                degenerate.append(row)
    if len(episodes) != args.expected or errors or degenerate:
        examples = [f"{row.get('episode_id')}: {row.get('error')}" for row in errors[:3]]
        raise SystemExit(
            f"invalid evaluation: episodes={len(episodes)}/{args.expected}, "
            f"errors={len(errors)}, degenerate_all_zero={len(degenerate)}; examples={examples}"
        )
    print(f"validated {len(episodes)} completed episodes: {args.jsonl}")


if __name__ == "__main__":
    main()
