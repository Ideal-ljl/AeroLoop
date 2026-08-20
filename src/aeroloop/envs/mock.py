"""Compatibility import for integrations using the legacy envs package."""

from ..simulators.mock import MockSimulator

MockEnvironment = MockSimulator

__all__ = ["MockEnvironment"]
