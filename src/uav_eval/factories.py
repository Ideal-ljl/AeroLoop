from __future__ import annotations

from typing import Any, Mapping

from .extensions import create_extension
from .metrics import Metric
from .envs.mock import MockEnvironment
from .policies.http import HttpPolicy
from .policies.mock import MockPolicy
from .policies.plugin import load_plugin
from .protocols import EnvironmentAdapter, PolicyAdapter


def build_policy(config: Mapping[str, Any]) -> PolicyAdapter:
    kind = str(config.get("type", "mock"))
    kwargs = dict(config.get("kwargs", {}))
    if kind == "mock":
        return MockPolicy(**kwargs)
    if kind == "http":
        return HttpPolicy(**kwargs)
    if kind == "plugin":
        return load_plugin(str(config["entrypoint"]), kwargs)
    if kind == "ckpt_harness":
        from .policies.harness import CkptHarnessPolicy

        return CkptHarnessPolicy(**kwargs)
    extension = str(config.get("entrypoint", kind))
    policy = create_extension(extension, "uav_eval.policies", kwargs)
    for method in ("reset", "predict"):
        if not callable(getattr(policy, method, None)):
            raise TypeError(f"policy extension {extension!r} is missing {method}()")
    return policy


def build_environment(config: Mapping[str, Any], env_name: str) -> EnvironmentAdapter:
    kind = str(config.get("type", "mock"))
    kwargs = dict(config.get("kwargs", {}))
    if kind == "mock":
        return MockEnvironment(**kwargs)
    if kind == "airbrain":
        from .envs.airbrain import AirBrainEnvironment

        return AirBrainEnvironment(env_name=env_name, **kwargs)
    extension = str(config.get("entrypoint", kind))
    environment = create_extension(extension, "uav_eval.environments", {"env_name": env_name, **kwargs})
    for method in ("reset", "execute"):
        if not callable(getattr(environment, method, None)):
            raise TypeError(f"environment extension {extension!r} is missing {method}()")
    return environment


def build_custom_metrics(configs: list[Mapping[str, Any]] | None) -> list[Metric]:
    metrics = []
    for config in configs or []:
        name = str(config.get("type") or config.get("entrypoint") or "")
        if not name:
            raise ValueError("custom metric requires type or entrypoint")
        metric = create_extension(name, "uav_eval.metrics", config.get("kwargs", {}))
        for method in ("reset", "finalize"):
            if not callable(getattr(metric, method, None)):
                raise TypeError(f"metric extension {name!r} is missing {method}()")
        metrics.append(metric)
    return metrics
