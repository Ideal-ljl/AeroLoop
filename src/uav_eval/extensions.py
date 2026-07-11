from __future__ import annotations

import importlib
from importlib import metadata
from typing import Any, Mapping


def import_object(path: str):
    if ":" not in path:
        raise ValueError("extension path must be 'python.module:ObjectName'")
    module_name, object_name = path.split(":", 1)
    return getattr(importlib.import_module(module_name), object_name)


def resolve_extension(name_or_path: str, group: str):
    """Resolve an explicit import path or an installed package entry point."""

    if ":" in name_or_path:
        return import_object(name_or_path)
    candidates = metadata.entry_points()
    selected = candidates.select(group=group, name=name_or_path)
    matches = list(selected)
    if not matches:
        raise KeyError(
            f"unknown extension {name_or_path!r} in {group!r}; use a module:Object path "
            "or install a package exposing that entry-point group"
        )
    if len(matches) > 1:
        raise RuntimeError(f"multiple extensions named {name_or_path!r} are installed in {group!r}")
    return matches[0].load()


def create_extension(name_or_path: str, group: str, kwargs: Mapping[str, Any] | None = None):
    factory = resolve_extension(name_or_path, group)
    return factory(**dict(kwargs or {}))
