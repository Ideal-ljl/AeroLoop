from __future__ import annotations

import importlib
from typing import Any, Mapping

from ..protocols import PolicyAdapter


def load_plugin(entrypoint: str, kwargs: Mapping[str, Any] | None = None) -> PolicyAdapter:
    if ":" not in entrypoint:
        raise ValueError("plugin entrypoint must be 'python.module:ClassName'")
    module_name, class_name = entrypoint.split(":", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    instance = cls(**dict(kwargs or {}))
    for method in ("reset", "predict"):
        if not callable(getattr(instance, method, None)):
            raise TypeError(f"plugin {entrypoint} is missing callable {method}()")
    return instance
