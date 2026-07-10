from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

from .config import load_config
from .episodes import load_airbrain_episodes, load_inline_episodes
from .factories import build_environment, build_policy
from .metrics import MetricConfig, aggregate_results
from .recording import JsonlRecorder
from .runner import RolloutConfig, RolloutRunner


def _load_episodes(config: dict):
    benchmark = config.get("benchmark", {})
    source = benchmark.get("source", "inline")
    if source == "airbrain":
        return load_airbrain_episodes(
            benchmark["eval_config"],
            repo_id=benchmark.get("repo_id"),
            max_samples=benchmark.get("max_samples"),
        )
    if source == "inline":
        return load_inline_episodes(benchmark.get("episodes", []))
    raise KeyError(f"unknown benchmark source: {source}")


def run(config: dict) -> Path:
    episodes = _load_episodes(config)
    if not episodes:
        raise ValueError("benchmark selected zero episodes")
    policy = build_policy(config.get("policy", {"type": "mock"}))
    rollout = RolloutConfig(**config.get("rollout", {}))
    metric_config = MetricConfig(**config.get("metrics", {}))
    output_cfg = config.get("output", {})
    output_path = Path(output_cfg.get("jsonl", f"eval_results/run_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"))
    include_steps = bool(output_cfg.get("include_steps", True))
    recorder = JsonlRecorder(output_path)
    recorder.append({"type": "run_config", "config": config})

    grouped = defaultdict(list)
    for episode in episodes:
        grouped[episode.env_name].append(episode)

    all_metric_rows = []
    try:
        for env_name, env_episodes in grouped.items():
            environment = build_environment(config.get("environment", {"type": "mock"}), env_name)
            env_rows = []
            try:
                runner = RolloutRunner(environment, policy, rollout, metric_config)
                for index, episode in enumerate(env_episodes, start=1):
                    print(f"[{env_name}] episode {index}/{len(env_episodes)}: {episode.episode_id}")
                    result = runner.run_episode(episode)
                    recorder.append(result.as_dict(include_steps=include_steps))
                    row = dict(result.metrics)
                    env_rows.append(row)
                    all_metric_rows.append(row)
                    print(
                        f"  termination={result.termination_reason.value} "
                        f"success={row['success']} distance={row['distance_to_goal']:.3f} steps={row['steps_taken']}"
                    )
            finally:
                environment.close()
            env_summary = aggregate_results(env_rows)
            env_summary.update({"type": "environment_summary", "env_name": env_name})
            recorder.append(env_summary)
    finally:
        policy.close()

    overall = aggregate_results(all_metric_rows)
    overall["type"] = "overall_summary"
    overall["output_file"] = str(output_path)
    recorder.append(overall)
    print(f"completed {len(all_metric_rows)} episodes; results: {output_path}")
    return output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Model-agnostic UAV navigation evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="run a benchmark config")
    run_parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        run(load_config(args.config))


if __name__ == "__main__":
    main()
