#!/usr/bin/env python3
"""Recover the fixed OpenFly 11x100 manifest from completed evaluation logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ENVIRONMENTS = (
    "env_airsim_16",
    "env_airsim_18",
    "env_airsim_26",
    "env_airsim_sh",
    "env_gs_ecust",
    "env_gs_nwpu01",
    "env_gs_nwpu02",
    "env_gs_sjtu01",
    "env_gs_sjtu02",
    "env_ue_bigcity",
    "env_ue_smallcity",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-jsonl", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    recovered: dict[tuple[str, int], dict[str, str]] = {}
    for source in args.source_jsonl:
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                env_name = row.get("env_name")
                sample_idx = row.get("sample_idx")
                path = row.get("path")
                instruction = row.get("instruction")
                if (
                    env_name not in ENVIRONMENTS
                    or not isinstance(sample_idx, int)
                    or not 0 <= sample_idx < 100
                    or not path
                    or not instruction
                ):
                    continue
                key = (env_name, sample_idx)
                candidate = {"path": str(path), "instruction": str(instruction)}
                previous = recovered.setdefault(key, candidate)
                if previous != candidate:
                    raise ValueError(f"conflicting recovered rows for {key}")

    missing = [
        (env_name, sample_idx)
        for env_name in ENVIRONMENTS
        for sample_idx in range(100)
        if (env_name, sample_idx) not in recovered
    ]
    if missing:
        raise ValueError(f"missing {len(missing)} rows; first missing keys: {missing[:10]}")

    rows = [
        recovered[(env_name, sample_idx)]
        for env_name in ENVIRONMENTS
        for sample_idx in range(100)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"recovered {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
