from __future__ import annotations

import argparse
import shutil
import time
import importlib.util
from collections import defaultdict
from importlib import resources
from pathlib import Path

from .config import load_config
from .datasets import load_openfly_episodes, load_traveluav_episodes, write_dataset_html
from .episodes import load_inline_episodes
from .extensions import resolve_extension
from .factories import build_custom_metrics, build_observers, build_policy, build_simulator
from .metrics import MetricConfig, aggregate_results
from .media import MediaConfig, MediaObserver
from .recording import JsonlRecorder
from .runner import RolloutConfig, RolloutRunner
from . import __version__


def _load_episodes(config: dict):
    benchmark = config.get("benchmark", {})
    source = benchmark.get("source", "inline")
    if source == "inline":
        return load_inline_episodes(benchmark.get("episodes", []))
    if source == "openfly":
        return load_openfly_episodes(**dict(benchmark.get("kwargs", {})))
    if source == "traveluav":
        return load_traveluav_episodes(**dict(benchmark.get("kwargs", {})))
    loader = resolve_extension(str(benchmark.get("entrypoint", source)), "aeroloop.episode_sources")
    episodes = loader(**dict(benchmark.get("kwargs", {})))
    return list(episodes)


def run(config: dict) -> Path:
    episodes = _load_episodes(config)
    if not episodes:
        raise ValueError("benchmark selected zero episodes")
    policy = build_policy(config.get("policy", {"type": "mock"}))
    rollout = RolloutConfig(**config.get("rollout", {}))
    metrics_cfg = dict(config.get("metrics", {}))
    custom_metrics = build_custom_metrics(metrics_cfg.pop("custom", []))
    metric_config = MetricConfig.from_mapping(metrics_cfg)
    output_cfg = config.get("output", {})
    output_path = Path(output_cfg.get("jsonl", f"eval_results/run_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"))
    include_steps = bool(output_cfg.get("include_steps", True))
    recorder = JsonlRecorder(output_path)
    recorder.append({"type": "run_config", "config": config})
    media_config = MediaConfig(**config.get("media", {}))
    observers = build_observers(config.get("observers", []))
    if media_config.show_window or media_config.save_video or media_config.save_collision_frame:
        observers.append(MediaObserver(media_config))

    grouped = defaultdict(list)
    for episode in episodes:
        grouped[episode.env_name].append(episode)

    all_metric_rows = []
    try:
        for env_name, env_episodes in grouped.items():
            simulator_config = config.get("simulator", config.get("environment", {"type": "mock"}))
            simulator = build_simulator(simulator_config, env_name)
            env_rows = []
            try:
                runner = RolloutRunner(
                    simulator,
                    policy,
                    rollout,
                    metric_config,
                    observers=observers,
                    custom_metrics=custom_metrics,
                )
                for index, episode in enumerate(env_episodes, start=1):
                    print(f"[{env_name}] episode {index}/{len(env_episodes)}: {episode.episode_id}")
                    result = runner.run_episode(episode)
                    result_row = result.as_dict(include_steps=include_steps)
                    labels = {
                        key: episode.metadata[key]
                        for key in ("benchmark_split", "difficulty", "source_split")
                        if key in episode.metadata
                    }
                    result_row.update(labels)
                    recorder.append(result_row)
                    row = {**dict(result.metrics), **labels}
                    env_rows.append(row)
                    all_metric_rows.append(row)
                    print(
                        f"  termination={result.termination_reason.value} "
                        f"success={row['success']} distance={row['distance_to_goal']:.3f} steps={row['steps_taken']}"
                    )
            finally:
                simulator.close()
            env_summary = aggregate_results(env_rows)
            env_summary.update({"type": "environment_summary", "env_name": env_name})
            recorder.append(env_summary)
    finally:
        policy.close()
        for observer in observers:
            observer.close()

    overall = aggregate_results(all_metric_rows)
    overall["type"] = "overall_summary"
    overall["output_file"] = str(output_path)
    recorder.append(overall)
    split_names = sorted(
        {str(row["benchmark_split"]) for row in all_metric_rows if row.get("benchmark_split") is not None}
    )
    for split_name in split_names:
        split_rows = [row for row in all_metric_rows if row.get("benchmark_split") == split_name]
        for difficulty in ("full", "easy", "hard"):
            selected = (
                split_rows
                if difficulty == "full"
                else [row for row in split_rows if row.get("difficulty") == difficulty]
            )
            if not selected:
                continue
            summary = aggregate_results(selected)
            summary.update(
                {
                    "type": "benchmark_group_summary",
                    "benchmark_split": split_name,
                    "difficulty": difficulty,
                }
            )
            recorder.append(summary)
    print(f"completed {len(all_metric_rows)} episodes; results: {output_path}")
    return output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="UAV simulator-model interaction and data collection")
    parser.add_argument("--version", action="version", version=f"aeroloop {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="run an interaction config")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--output-jsonl", help="override output.jsonl")
    run_parser.add_argument("--headless", action="store_true", help="disable the visualization window")
    run_parser.add_argument("--no-video", action="store_true", help="disable MP4 recording")
    sub.add_parser("doctor", help="check optional runtime dependencies")
    init_parser = sub.add_parser("init-config", help="write a packaged config template")
    init_parser.add_argument("--template", choices=("mock", "http"), default="mock")
    init_parser.add_argument("--output", default="aeroloop.yaml")
    inspect_parser = sub.add_parser("inspect-dataset", help="validate and summarize a configured dataset")
    inspect_parser.add_argument("--config", required=True)
    inspect_parser.add_argument("--limit", type=int, default=5)
    visualize_parser = sub.add_parser("visualize-dataset", help="write an interactive-free HTML trajectory preview")
    visualize_parser.add_argument("--config", required=True)
    visualize_parser.add_argument("--output", default="eval_results/dataset.html")
    visualize_parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args(argv)
    if args.command == "run":
        config = load_config(args.config)
        if args.output_jsonl is not None:
            config.setdefault("output", {})["jsonl"] = args.output_jsonl
        if args.headless:
            config.setdefault("media", {})["show_window"] = False
        if args.no_video:
            config.setdefault("media", {})["save_video"] = False
        run(config)
    elif args.command == "doctor":
        checks = {
            "numpy": importlib.util.find_spec("numpy") is not None,
            "cv2": importlib.util.find_spec("cv2") is not None,
            "airsim": importlib.util.find_spec("airsim") is not None,
            "airsim_ue5": importlib.util.find_spec("airsim_ue5") is not None,
            "unrealcv": importlib.util.find_spec("unrealcv") is not None,
            "ffmpeg": shutil.which("ffmpeg") is not None,
        }
        for name, available in checks.items():
            print(f"{name:8s} {'OK' if available else 'MISSING'}")
    elif args.command == "init-config":
        destination = Path(args.output)
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite existing config: {destination}")
        template = (
            resources.files("aeroloop").joinpath("resources", f"{args.template}.yaml").read_text(encoding="utf-8")
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(template, encoding="utf-8")
        print(f"wrote {args.template} template: {destination}")
    elif args.command == "inspect-dataset":
        episodes = _load_episodes(load_config(args.config))
        print(f"episodes={len(episodes)} environments={len({episode.env_name for episode in episodes})}")
        for episode in episodes[: args.limit]:
            print(
                f"{episode.episode_id} env={episode.env_name} path={episode.reference_path_length:.2f} "
                f"start={episode.start_pose.as_list()} target={list(episode.target_position)}"
            )
    elif args.command == "visualize-dataset":
        episodes = _load_episodes(load_config(args.config))
        path = write_dataset_html(episodes, args.output, args.limit)
        print(f"wrote dataset preview: {path}")


if __name__ == "__main__":
    main()
