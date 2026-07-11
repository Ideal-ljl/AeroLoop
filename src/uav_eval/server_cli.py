from __future__ import annotations

import argparse

from .server.httpd import serve


def _network_args(parser):
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18080)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve a UAV model through the UAVEval HTTP contract")
    sub = parser.add_subparsers(dest="backend", required=True)

    aerial = sub.add_parser("aerialvla")
    _network_args(aerial)
    aerial.add_argument("--repo-root", required=True)
    aerial.add_argument("--ckpt-dir", required=True)
    aerial.add_argument("--device", default="cuda")
    aerial.add_argument("--dtype", default="bfloat16")

    openuav = sub.add_parser("openuav")
    _network_args(openuav)
    openuav.add_argument("--repo-root", required=True)
    openuav.add_argument("--ckpt-dir", required=True)
    openuav.add_argument("--device", default="cuda")
    openuav.add_argument("--stop-norm-threshold", type=float, default=0.5)
    openuav.add_argument("--history-size", type=int, default=4)

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

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    common = {"host": args.host, "port": args.port}
    if args.backend == "aerialvla":
        from .server.backends.aerialvla import AerialVLABackend

        backend = AerialVLABackend(args.repo_root, args.ckpt_dir, args.device, args.dtype)
    elif args.backend == "openuav":
        from .server.backends.openuav import OpenUAVBackend

        backend = OpenUAVBackend(
            args.repo_root, args.ckpt_dir, args.device, args.stop_norm_threshold, args.history_size
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
    elif args.backend == "worldvln":
        from .server.backends.worldvln import WorldVLNProxyBackend

        backend = WorldVLNProxyBackend(
            upstream_url=args.upstream_url,
            timeout_s=args.timeout_s,
            action_head_mode=args.action_head_mode,
            stop_speed_threshold=args.stop_speed_threshold,
        )
    else:  # pragma: no cover
        raise KeyError(args.backend)
    serve(backend, **common)


if __name__ == "__main__":
    main()
