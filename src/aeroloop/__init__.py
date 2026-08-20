"""AeroLoop public API."""

__version__ = "0.1.0"

from .metrics import Metric, MetricConfig
from .cameras import CameraSpec, STANDARD_CAMERAS, resolve_cameras
from .media import MediaConfig, MediaObserver
from .protocols import EnvironmentAdapter, PolicyAdapter, SimulatorAdapter
from .observers import RolloutObserver
from .runner import RolloutConfig, RolloutRunner
from .simulators.mock import MockSimulator
from .simulators import AirSimSimulator, GSAirSimSimulator, UnrealCVSimulator
from .types import ActionChunk, CanonicalAction, EpisodeResult, EpisodeSpec, Observation, Pose
from .datasets import load_openfly_episodes, load_traveluav_episodes, write_dataset_html

__all__ = [
    "ActionChunk",
    "AirSimSimulator",
    "CanonicalAction",
    "CameraSpec",
    "EnvironmentAdapter",
    "EpisodeResult",
    "EpisodeSpec",
    "MediaConfig",
    "MediaObserver",
    "Metric",
    "MetricConfig",
    "MockSimulator",
    "GSAirSimSimulator",
    "Observation",
    "PolicyAdapter",
    "Pose",
    "RolloutConfig",
    "RolloutObserver",
    "RolloutRunner",
    "SimulatorAdapter",
    "UnrealCVSimulator",
    "STANDARD_CAMERAS",
    "resolve_cameras",
    "load_openfly_episodes",
    "load_traveluav_episodes",
    "write_dataset_html",
]
