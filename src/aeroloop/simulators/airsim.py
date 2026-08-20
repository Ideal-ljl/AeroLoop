from __future__ import annotations

import importlib
import math
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..cameras import CameraSpec, resolve_cameras
from ..geometry import apply_body_action, relative_state
from ..protocols import SimulatorAdapter
from ..types import CanonicalAction, EpisodeSpec, Observation, Pose, Transition
from .process import find_file, launch_process, scene_directory, set_json_value, stop_process, wait_for_port


def _triple(name: str, values: Sequence[float]) -> tuple[float, float, float]:
    row = tuple(map(float, values))
    if len(row) != 3 or not all(math.isfinite(value) and value != 0 for value in row):
        raise ValueError(f"{name} must contain three finite non-zero values")
    return row


@dataclass
class AirSimSimulator(SimulatorAdapter):
    """Direct AirSim-compatible simulator control without an external bridge."""

    env_name: str
    env_root: str
    cameras: Any = "front"
    camera_names: Mapping[str, str] = field(default_factory=lambda: {"front": "front_custom"})
    host: str = "127.0.0.1"
    port: int = 41451
    sdk_module: str = "airsim"
    client_class: str = "MultirotorClient"
    launch_script: str = "LinuxNoEditor/start.sh"
    launch_args: Sequence[str] | str = ()
    launch: bool = True
    settings_filename: str = "settings.json"
    update_settings_port: bool = True
    startup_timeout_s: float = 120.0
    settle_time_s: float = 0.05
    position_sign: Sequence[float] = (1.0, -1.0, -1.0)
    position_scale: float = 1.0
    yaw_sign: float = -1.0
    yaw_offset: float = 0.0
    image_hflip: bool = False
    channel_order: str = "rgb"
    ignore_collision: bool = False
    vehicle_name: str = ""
    enable_api_control: bool = True
    arm: bool = True
    configure_camera_poses: bool = False
    strict_camera_size: bool = False
    capture_depth: bool = False
    depth_image_type: str = "DepthPlanar"
    log_path: str | None = None
    sdk: Any = field(default=None, repr=False)
    name: str = "airsim"

    def __post_init__(self) -> None:
        self.camera_specs = resolve_cameras(self.cameras)
        self.camera_names = {str(key): str(value) for key, value in self.camera_names.items()}
        self.position_sign = _triple("position_sign", self.position_sign)
        self.position_scale = float(self.position_scale)
        if not math.isfinite(self.position_scale) or self.position_scale <= 0:
            raise ValueError("position_scale must be a finite positive number")
        if self.channel_order not in {"rgb", "bgr"}:
            raise ValueError("channel_order must be rgb or bgr")
        self.process = None
        self.client = None
        self.scene_dir: Path | None = None
        self._owns_process = False

    def _sdk(self):
        if self.sdk is None:
            try:
                self.sdk = importlib.import_module(self.sdk_module)
            except ImportError as exc:
                raise RuntimeError(
                    f"{self.name} requires the optional simulator SDK {self.sdk_module!r} in this environment"
                ) from exc
        return self.sdk

    def _start(self) -> None:
        if self.client is not None:
            return
        sdk = self._sdk()
        if self.launch:
            self.scene_dir = scene_directory(self.env_root, self.env_name)
            if self.update_settings_port:
                settings = find_file(self.scene_dir, self.settings_filename)
                set_json_value(settings, "ApiServerPort", int(self.port))
            script = self.scene_dir / self.launch_script
            if not script.is_file():
                raise FileNotFoundError(f"simulator launch script not found: {script}")
            args = shlex.split(self.launch_args) if isinstance(self.launch_args, str) else list(self.launch_args)
            self.process = launch_process(
                ["bash", str(script), *args], cwd=self.scene_dir, log_path=self.log_path
            )
            self._owns_process = True
            wait_for_port(self.host, self.port, self.startup_timeout_s, self.process)
        client_type = getattr(sdk, self.client_class)
        try:
            self.client = client_type(ip=self.host, port=int(self.port))
        except TypeError:
            self.client = client_type(self.host, int(self.port))
        if callable(getattr(self.client, "confirmConnection", None)):
            self.client.confirmConnection()
        if self.enable_api_control and callable(getattr(self.client, "enableApiControl", None)):
            self.client.enableApiControl(True, self.vehicle_name)
        if self.arm and callable(getattr(self.client, "armDisarm", None)):
            self.client.armDisarm(True, self.vehicle_name)
        self._configure_cameras()

    def _native_pose(self, pose: Pose):
        sdk = self._sdk()
        x, y, z = (
            value * self.position_scale * sign
            for value, sign in zip(pose.xyz(), self.position_sign)
        )
        orientation = sdk.to_quaternion(0.0, 0.0, pose.yaw * float(self.yaw_sign) + float(self.yaw_offset))
        return sdk.Pose(sdk.Vector3r(x, y, z), orientation)

    def _camera_pose(self, camera: CameraSpec):
        sdk = self._sdk()
        x, y, z = (
            value * self.position_scale * sign
            for value, sign in zip(camera.position, self.position_sign)
        )
        orientation = sdk.to_quaternion(
            camera.pitch,
            camera.roll,
            camera.yaw * float(self.yaw_sign),
        )
        return sdk.Pose(sdk.Vector3r(x, y, z), orientation)

    def _configure_cameras(self) -> None:
        if not self.configure_camera_poses:
            return
        setter = getattr(self.client, "simSetCameraPose", None)
        if not callable(setter):
            return
        for camera in self.camera_specs:
            native_name = self.camera_names.get(camera.name, camera.name)
            try:
                setter(native_name, self._camera_pose(camera), self.vehicle_name)
            except TypeError:
                setter(native_name, self._camera_pose(camera))

    def _set_pose(self, pose: Pose, *, ignore_collision: bool) -> None:
        native = self._native_pose(pose)
        try:
            self.client.simSetVehiclePose(native, bool(ignore_collision), self.vehicle_name)
        except TypeError:
            self.client.simSetVehiclePose(native, bool(ignore_collision))
        if self.settle_time_s > 0:
            time.sleep(self.settle_time_s)

    def _decode_image(self, response):
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("AirSim image capture requires NumPy") from exc
        if getattr(response, "pixels_as_float", False):
            raise ValueError("scene camera unexpectedly returned floating-point pixels")
        height, width = int(response.height), int(response.width)
        raw = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        channels = raw.size // max(height * width, 1)
        if channels not in (3, 4) or raw.size != height * width * channels:
            raise ValueError(f"unexpected AirSim image buffer: {raw.size} bytes for {width}x{height}")
        image = raw.reshape(height, width, channels)[..., :3]
        if self.channel_order == "bgr":
            image = image[..., ::-1]
        if self.image_hflip:
            image = image[:, ::-1]
        return np.ascontiguousarray(image)

    def _decode_depth(self, response):
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("AirSim depth capture requires NumPy") from exc
        if not getattr(response, "pixels_as_float", False):
            raise ValueError("depth camera unexpectedly returned byte pixels")
        height, width = int(response.height), int(response.width)
        depth = np.asarray(response.image_data_float, dtype=np.float32)
        if depth.size != height * width:
            raise ValueError(f"unexpected AirSim depth buffer: {depth.size} values for {width}x{height}")
        return np.ascontiguousarray(depth.reshape(height, width))

    def _capture(self) -> tuple[dict[str, Any], dict[str, Any]]:
        sdk = self._sdk()
        requests = []
        request_rows = []
        for camera in self.camera_specs:
            native_name = self.camera_names.get(camera.name, camera.name)
            requests.append(sdk.ImageRequest(native_name, sdk.ImageType.Scene, False, False))
            request_rows.append((camera, "rgb"))
            if self.capture_depth:
                try:
                    image_type = getattr(sdk.ImageType, self.depth_image_type)
                except AttributeError as exc:
                    raise ValueError(f"AirSim SDK has no image type {self.depth_image_type!r}") from exc
                requests.append(sdk.ImageRequest(native_name, image_type, True, False))
                request_rows.append((camera, "depth"))
        try:
            responses = self.client.simGetImages(requests, self.vehicle_name)
        except TypeError:
            responses = self.client.simGetImages(requests)
        if len(responses) != len(requests):
            raise RuntimeError(f"AirSim returned {len(responses)} images for {len(requests)} requests")
        images, depths = {}, {}
        for (camera, kind), response in zip(request_rows, responses):
            value = self._decode_image(response) if kind == "rgb" else self._decode_depth(response)
            if self.strict_camera_size:
                expected = (camera.height, camera.width)
                actual = value.shape[:2]
                if all(item is not None for item in expected) and actual != expected:
                    raise ValueError(f"camera {camera.name!r} returned {actual}, expected {expected}")
            (images if kind == "rgb" else depths)[camera.name] = value
        return images, depths

    def _collision(self) -> tuple[bool, dict[str, Any]]:
        getter = getattr(self.client, "simGetCollisionInfo", None)
        if not callable(getter):
            return False, {}
        info = getter(self.vehicle_name) if self.vehicle_name else getter()
        collided = bool(getattr(info, "has_collided", False))
        details = {
            "object_name": str(getattr(info, "object_name", "")),
            "collision_timestamp": int(getattr(info, "time_stamp", 0)),
        }
        return collided, details

    def _observation(self) -> Observation:
        images, depths = self._capture()
        return Observation(
            rgb=images[self.camera_specs[0].name],
            images=images,
            primary_view=self.camera_specs[0].name,
            camera_specs={camera.name: camera for camera in self.camera_specs},
            pose=self.pose,
            relative_state=relative_state(self.pose, self.origin),
            step_index=self.step,
            info={"depth": depths} if depths else {},
            auxiliary_state={"coordinate_frame": "aeroloop", "native_sdk": self.sdk_module},
        )

    def reset(self, episode: EpisodeSpec) -> Observation:
        self._start()
        self.episode = episode
        self.origin = episode.start_pose
        self.pose = episode.start_pose
        self.step = 0
        self._set_pose(self.pose, ignore_collision=True)
        _, collision_info = self._collision()
        self._collision_baseline_timestamp = collision_info.get("collision_timestamp", 0)
        return self._observation()

    def execute(self, action: CanonicalAction) -> Transition:
        # Rendering and detector inference can take long enough for the native
        # vehicle physics to report an idle-period collision even though
        # AeroLoop owns the benchmark pose.  Snapshot that state before the
        # kinematic move and only treat a newly timestamped collision as an
        # action collision.
        _, idle_collision_info = self._collision()
        idle_collision_timestamp = idle_collision_info.get("collision_timestamp", 0)
        self.pose = apply_body_action(self.pose, action)
        self._set_pose(self.pose, ignore_collision=self.ignore_collision)
        self.step += 1
        collision, collision_info = self._collision()
        timestamp = collision_info.get("collision_timestamp", 0)
        stale_timestamps = {
            idle_collision_timestamp,
            getattr(self, "_collision_baseline_timestamp", 0),
        }
        if collision and timestamp and timestamp in stale_timestamps:
            collision = False
            collision_info["ignored_stale_collision"] = True
        observation = self._observation()
        return Transition(observation, collision=collision, info={"collision": collision_info})

    def close(self) -> None:
        if self.client is not None:
            if self.arm and callable(getattr(self.client, "armDisarm", None)):
                try:
                    self.client.armDisarm(False, self.vehicle_name)
                except Exception:
                    pass
            if self.enable_api_control and callable(getattr(self.client, "enableApiControl", None)):
                try:
                    self.client.enableApiControl(False, self.vehicle_name)
                except Exception:
                    pass
        if self._owns_process:
            stop_process(self.process)
        self.client = None
        self.process = None


class GSAirSimSimulator(AirSimSimulator):
    """Preset for the UE5/GS AirSim-compatible SDK used by GS scenes."""

    def __init__(self, **kwargs):
        defaults = {
            "sdk_module": "airsim_ue5",
            "client_class": "VehicleClient",
            "launch_script": "gs.sh",
            "position_sign": (1.0, 1.0, -1.0),
            "yaw_sign": 1.0,
            "image_hflip": True,
            "arm": False,
            "name": "gs_airsim",
        }
        super().__init__(**{**defaults, **kwargs})
