#!/usr/bin/env python3
"""Summarize complete or in-progress UAVEval AirBrain JSONL directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean


FIELDS = ("success", "spl", "osr", "collision", "distance_to_goal")


def summarize(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    run = next((row for row in rows if row.get("type") == "run_config"), {})
    episodes = [row for row in rows if row.get("type") == "episode"]
    overall = next((row for row in reversed(rows) if row.get("type") == "overall_summary"), None)
    benchmark = run.get("config", {}).get("benchmark", {})
    policy_url = run.get("config", {}).get("policy", {}).get("kwargs", {}).get("url", "")
    port_to_model = {"18101": "aerialvla", "18103": "dualvln", "18104": "worldvln"}
    model = next((name for port, name in port_to_model.items() if port in policy_url), path.parent.name.split("_", 1)[0])
    errors = [row for row in episodes if row.get("termination_reason") == "error" or row.get("error")]
    expected = int(benchmark.get("max_samples") or 0)
    result = {
        "model": model,
        "environment": benchmark.get("repo_id") or (episodes[0].get("env_name") if episodes else ""),
        "status": "complete" if overall and len(episodes) == expected and not errors else "failed" if errors else "running",
        "episodes": len(episodes),
        "expected": expected,
        "errors": len(errors),
        "total_steps": sum(int(row.get("steps_taken") or 0) for row in episodes),
        "inference_calls": sum(int(row.get("inference_calls") or 0) for row in episodes),
        "path": str(path),
    }
    for field in FIELDS:
        if overall and field in overall:
            result[field] = float(overall[field])
        else:
            values = [float(row[field]) for row in episodes if row.get(field) is not None]
            result[field] = fmean(values) if values else None
    return result


def markdown(rows: list[dict]) -> str:
    headers = ["model", "environment", "status", "episodes", "errors", "success", "spl", "osr", "collision", "distance"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        values = [
            row["model"], row["environment"], row["status"], f"{row['episodes']}/{row['expected']}", str(row["errors"]),
            *("-" if row[key] is None else f"{row[key]:.4f}" for key in ("success", "spl", "osr", "collision")),
            "-" if row["distance_to_goal"] is None else f"{row['distance_to_goal']:.3f}",
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--glob", default="*/eval_results.jsonl")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    args = parser.parse_args()
    paths = sorted(args.root.glob(args.glob))
    rows = sorted((summarize(path) for path in paths), key=lambda row: (row["model"], row["environment"], row["path"]))
    if args.format == "json":
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        print(markdown(rows))


if __name__ == "__main__":
    main()
