from __future__ import annotations

import argparse

from .config import load_config
from .server.httpd import serve


def _network_args(parser):
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18080)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve a UAV model through the AeroLoop HTTP contract")
    sub = parser.add_subparsers(dest="backend", required=True)

    aerial = sub.add_parser("aerialvla")
    _network_args(aerial)
    aerial.add_argument("--repo-root", required=True)
    aerial.add_argument("--ckpt-dir", required=True)
    aerial.add_argument("--adapter-dir")
    aerial.add_argument("--device", default="cuda")
    aerial.add_argument("--dtype", default="bfloat16")

    openuav = sub.add_parser("openuav")
    _network_args(openuav)
    openuav.add_argument("--repo-root", required=True)
    openuav.add_argument("--ckpt-dir", required=True)
    openuav.add_argument("--device", default="cuda")
    openuav.add_argument("--stop-norm-threshold", type=float, default=0.5)
    openuav.add_argument("--history-size", type=int, default=4)
    openuav.add_argument("--traj-model-path")

    dual = sub.add_parser("dualvln")
    _network_args(dual)
    dual.add_argument("--repo-root", required=True)
    dual.add_argument("--ckpt-dir", required=True)
    dual.add_argument("--device", default="cuda")
    dual.add_argument("--dtype", default="bfloat16")
    dual.add_argument("--predict-steps", type=int, default=32)
    dual.add_argument("--inference-steps", type=int, default=10)
    dual.add_argument("--sample-trajectories", type=int, default=32)
    dual.add_argument("--stop-speed-threshold", type=float, default=0.05)
    dual.add_argument("--seed", type=int, default=2026)

    omninav = sub.add_parser("omninav")
    _network_args(omninav)
    omninav.add_argument("--repo-root", required=True)
    omninav.add_argument("--ckpt-dir", required=True)
    omninav.add_argument("--device", default="cuda")
    omninav.add_argument("--dtype", default="bfloat16")
    omninav.add_argument("--attn-impl", default="sdpa")
    omninav.add_argument("--history-size", type=int, default=5)
    omninav.add_argument("--current-long-edge", type=int, default=640)

    pi0 = sub.add_parser("pi0")
    _network_args(pi0)
    pi0.add_argument("--repo-root", required=True)
    pi0.add_argument("--ckpt-dir", required=True)
    pi0.add_argument("--tokenizer-dir", required=True)
    pi0.add_argument("--device", default="cuda")
    pi0.add_argument("--dtype", default="bfloat16")
    pi0.add_argument("--seed", type=int, default=2026)
    pi0.add_argument("--inference-steps", type=int)

    world = sub.add_parser("worldvln")
    _network_args(world)
    world.add_argument("--upstream-url", default="http://127.0.0.1:8001")
    world.add_argument("--timeout-s", type=float, default=600)
    world.add_argument("--action-head-mode", default="tsformer_latent")
    world.add_argument(
        "--stop-speed-threshold",
        type=float,
        default=0.5,
        help="stop when the final three predicted translations average below this many metres",
    )

    function = sub.add_parser("function", help="wrap an existing Python inference function")
    _network_args(function)
    function.add_argument("--config", required=True, help="YAML mapping function arguments to protocol fields")

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    common = {"host": args.host, "port": args.port}
    if args.backend == "aerialvla":
        from .server.backends.aerialvla import AerialVLABackend

        backend = AerialVLABackend(
            args.repo_root,
            args.ckpt_dir,
            args.device,
            args.dtype,
            adapter_dir=args.adapter_dir,
        )
    elif args.backend == "openuav":
        from .server.backends.openuav import OpenUAVBackend

        backend = OpenUAVBackend(
            args.repo_root,
            args.ckpt_dir,
            args.device,
            args.stop_norm_threshold,
            args.history_size,
            args.traj_model_path,
        )
    elif args.backend == "dualvln":
        from .server.backends.dualvln import DualVLNBackend

        backend = DualVLNBackend(
            repo_root=args.repo_root,
            ckpt_dir=args.ckpt_dir,
            device=args.device,
            dtype=args.dtype,
            predict_steps=args.predict_steps,
            inference_steps=args.inference_steps,
            sample_trajectories=args.sample_trajectories,
            stop_speed_threshold=args.stop_speed_threshold,
            seed=args.seed,
        )
    elif args.backend == "omninav":
        from .server.backends.omninav import OmniNavBackend

        backend = OmniNavBackend(
            repo_root=args.repo_root,
            ckpt_dir=args.ckpt_dir,
            device=args.device,
            dtype=args.dtype,
            attn_impl=args.attn_impl,
            history_size=args.history_size,
            current_long_edge=args.current_long_edge,
        )
    elif args.backend == "pi0":
        from .server.backends.pi0 import PI0Backend

        backend = PI0Backend(
            repo_root=args.repo_root,
            ckpt_dir=args.ckpt_dir,
            tokenizer_dir=args.tokenizer_dir,
            device=args.device,
            dtype=args.dtype,
            seed=args.seed,
            inference_steps=args.inference_steps,
        )
    elif args.backend == "worldvln":
        from .server.backends.worldvln import WorldVLNProxyBackend

        backend = WorldVLNProxyBackend(
            upstream_url=args.upstream_url,
            timeout_s=args.timeout_s,
            action_head_mode=args.action_head_mode,
            stop_speed_threshold=args.stop_speed_threshold,
        )
    elif args.backend == "function":
        from .server.function import FunctionBackend

        config = load_config(args.config)
        backend = FunctionBackend(**dict(config.get("function", config)))
    else:  # pragma: no cover
        raise KeyError(args.backend)
    serve(backend, **common)


if __name__ == "__main__":
    main()
