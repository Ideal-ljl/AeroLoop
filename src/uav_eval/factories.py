from __future__ import annotations

from typing import Any, Mapping

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
    raise KeyError(f"unknown policy type: {kind}")


def build_environment(config: Mapping[str, Any], env_name: str) -> EnvironmentAdapter:
    kind = str(config.get("type", "mock"))
    kwargs = dict(config.get("kwargs", {}))
    if kind == "mock":
        return MockEnvironment(**kwargs)
    if kind == "airbrain":
        from .envs.airbrain import AirBrainEnvironment

        return AirBrainEnvironment(env_name=env_name, **kwargs)
    raise KeyError(f"unknown environment type: {kind}")
