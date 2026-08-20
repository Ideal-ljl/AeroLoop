from __future__ import annotations

from typing import Any, Mapping

from .extensions import create_extension, resolve_extension
from .metrics import Metric
from .observers import RolloutObserver
from .simulators.mock import MockSimulator
from .simulators.airsim import AirSimSimulator, GSAirSimSimulator
from .simulators.unrealcv import UnrealCVSimulator
from .policies.http import HttpPolicy
from .policies.mock import MockPolicy
from .policies.plugin import load_plugin
from .protocols import PolicyAdapter, SimulatorAdapter


def build_policy(config: Mapping[str, Any]) -> PolicyAdapter:
    kind = str(config.get("type", "mock"))
    kwargs = dict(config.get("kwargs", {}))
    if kind == "mock":
        return MockPolicy(**kwargs)
    if kind == "http":
        return HttpPolicy(**kwargs)
    if kind == "l1_oracle":
        from .policies.l1_oracle import L1OraclePolicy

        base_config = kwargs.pop("base")
        return L1OraclePolicy(base_policy=build_policy(base_config), **kwargs)
    if kind == "groundingdino_stop":
        from .policies.groundingdino import GroundingDinoStopPolicy

        base_config = kwargs.pop("base")
        return GroundingDinoStopPolicy(base_policy=build_policy(base_config), **kwargs)
    if kind == "plugin":
        return load_plugin(str(config["entrypoint"]), kwargs)
    if kind == "ckpt_harness":
        from .policies.harness import CkptHarnessPolicy

        return CkptHarnessPolicy(**kwargs)
    extension = str(config.get("entrypoint", kind))
    policy = create_extension(extension, "aeroloop.policies", kwargs)
    for method in ("reset", "predict"):
        if not callable(getattr(policy, method, None)):
            raise TypeError(f"policy extension {extension!r} is missing {method}()")
    return policy


def build_simulator(config: Mapping[str, Any], env_name: str) -> SimulatorAdapter:
    kind = str(config.get("type", "mock"))
    kwargs = dict(config.get("kwargs", {}))
    if kind == "mock":
        return MockSimulator(**kwargs)
    if kind == "airsim":
        return AirSimSimulator(env_name=env_name, **kwargs)
    if kind == "gs_airsim":
        return GSAirSimSimulator(env_name=env_name, **kwargs)
    if kind == "unrealcv":
        return UnrealCVSimulator(env_name=env_name, **kwargs)
    extension = str(config.get("entrypoint", kind))
    simulator_kwargs = {"env_name": env_name, **kwargs}
    if ":" in extension:
        simulator = create_extension(extension, "aeroloop.simulators", simulator_kwargs)
    else:
        try:
            factory = resolve_extension(extension, "aeroloop.simulators")
        except KeyError:
            # Compatibility with integrations using the legacy environment group.
            factory = resolve_extension(extension, "aeroloop.environments")
        simulator = factory(**simulator_kwargs)
    for method in ("reset", "execute"):
        if not callable(getattr(simulator, method, None)):
            raise TypeError(f"simulator extension {extension!r} is missing {method}()")
    return simulator


def build_environment(config: Mapping[str, Any], env_name: str) -> SimulatorAdapter:
    """Compatibility alias for integrations using the old environment name."""

    return build_simulator(config, env_name)


def build_custom_metrics(configs: list[Mapping[str, Any]] | None) -> list[Metric]:
    metrics = []
    for config in configs or []:
        name = str(config.get("type") or config.get("entrypoint") or "")
        if not name:
            raise ValueError("custom metric requires type or entrypoint")
        metric = create_extension(name, "aeroloop.metrics", config.get("kwargs", {}))
        for method in ("reset", "finalize"):
            if not callable(getattr(metric, method, None)):
                raise TypeError(f"metric extension {name!r} is missing {method}()")
        metrics.append(metric)
    return metrics


def build_observers(configs: list[Mapping[str, Any]] | None) -> list[RolloutObserver]:
    observers = []
    for config in configs or []:
        name = str(config.get("type") or config.get("entrypoint") or "")
        if not name:
            raise ValueError("observer requires type or entrypoint")
        observer = create_extension(name, "aeroloop.observers", config.get("kwargs", {}))
        for method in ("on_episode_start", "on_step", "on_episode_end", "artifacts", "close"):
            if not callable(getattr(observer, method, None)):
                raise TypeError(f"observer extension {name!r} is missing {method}()")
        observers.append(observer)
    return observers
