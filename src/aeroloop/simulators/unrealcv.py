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
from .process import launch_process, scene_directory, stop_process


@dataclass
class UnrealCVSimulator(SimulatorAdapter):
    """Direct UnrealCV camera control for packaged Unreal Engine scenes."""

    env_name: str
    env_root: str
    cameras: Any = "front"
    camera_ids: Mapping[str, int] = field(default_factory=lambda: {"front": 1})
    host: str = "127.0.0.1"
    port: int = 9000
    launch_script: str = "CitySample.sh"
    launch_args: Sequence[str] | str = ()
    launch: bool = True
    unrealcv_ini: str = "City_UE52/Binaries/Linux/unrealcv.ini"
    update_ini_port: bool = True
    spawn_cameras: bool = True
    startup_timeout_s: float = 120.0
    startup_grace_s: float = 0.0
    settle_time_s: float = 0.05
    position_scale: float = 100.0
    position_sign: Sequence[float] = (1.0, -1.0, 1.0)
    yaw_sign: float = -1.0
    collision_query: str | None = None
    log_path: str | None = None
    client: Any = field(default=None, repr=False)
    name: str = "unrealcv"

    def __post_init__(self) -> None:
        self.camera_specs = resolve_cameras(self.cameras)
        self.camera_ids = {str(key): int(value) for key, value in self.camera_ids.items()}
        self.position_sign = tuple(map(float, self.position_sign))
        if len(self.position_sign) != 3:
            raise ValueError("position_sign must contain three values")
        self.process = None
        self.scene_dir: Path | None = None
        self._owns_process = False

    def _start(self) -> None:
        if self.client is not None:
            return
        if self.launch:
            self.scene_dir = scene_directory(self.env_root, self.env_name)
            if self.update_ini_port:
                self._set_ini_port(self.scene_dir / self.unrealcv_ini)
            script = self.scene_dir / self.launch_script
            if not script.is_file():
                raise FileNotFoundError(f"UnrealCV launch script not found: {script}")
            args = shlex.split(self.launch_args) if isinstance(self.launch_args, str) else list(self.launch_args)
            self.process = launch_process(
                ["bash", str(script), *args], cwd=self.scene_dir, log_path=self.log_path
            )
            self._owns_process = True
            if self.startup_grace_s > 0:
                deadline = time.monotonic() + self.startup_grace_s
                while time.monotonic() < deadline:
                    if self.process.poll() is not None:
                        raise RuntimeError(
                            f"simulator process exited early with code {self.process.returncode}"
                        )
                    time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))
        try:
            client_type = importlib.import_module("unrealcv").Client
        except ImportError as exc:
            raise RuntimeError("UnrealCVSimulator requires the optional 'unrealcv' package") from exc
        self.client = client_type((self.host, int(self.port)))
        deadline = time.monotonic() + self.startup_timeout_s
        while True:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(f"simulator process exited early with code {self.process.returncode}")
            try:
                connected = self.client.connect(timeout=2)
            except TypeError:
                connected = self.client.connect()
            if connected:
                break
            if time.monotonic() >= deadline:
                raise ConnectionError(f"failed to connect to UnrealCV at {self.host}:{self.port}")
            time.sleep(1)
        if self.spawn_cameras:
            self._request("vset /cameras/spawn")
        self._configure_cameras()

    def _set_ini_port(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"UnrealCV configuration not found: {path}")
        lines = path.read_text(encoding="utf-8").splitlines()
        updated = []
        found = False
        for line in lines:
            if line.startswith("Port="):
                updated.append(f"Port={int(self.port)}")
                found = True
            else:
                updated.append(line)
        if not found:
            updated.append(f"Port={int(self.port)}")
        path.write_text("\n".join(updated) + "\n", encoding="utf-8")

    def _request(self, command: str):
        response = self.client.request(command)
        if isinstance(response, str) and response.lower().startswith("error"):
            raise RuntimeError(f"UnrealCV rejected {command!r}: {response}")
        return response

    def _configure_cameras(self) -> None:
        for camera in self.camera_specs:
            camera_id = self.camera_ids.get(camera.name)
            if camera_id is None:
                raise KeyError(f"no UnrealCV camera id configured for view {camera.name!r}")
            if camera.width is not None and camera.height is not None:
                self._request(f"vset /camera/{camera_id}/size {camera.width} {camera.height}")

    def _camera_world_pose(self, camera: CameraSpec) -> tuple[float, float, float, float, float, float]:
        cos_yaw, sin_yaw = math.cos(self.pose.yaw), math.sin(self.pose.yaw)
        offset_x = camera.position[0] * cos_yaw - camera.position[1] * sin_yaw
        offset_y = camera.position[0] * sin_yaw + camera.position[1] * cos_yaw
        x = (self.pose.x + offset_x) * self.position_scale * self.position_sign[0]
        y = (self.pose.y + offset_y) * self.position_scale * self.position_sign[1]
        z = (self.pose.z + camera.position[2]) * self.position_scale * self.position_sign[2]
        pitch = math.degrees(camera.pitch)
        yaw = math.degrees(self.pose.yaw + camera.yaw) * self.yaw_sign
        roll = math.degrees(camera.roll)
        return x, y, z, pitch, yaw, roll

    def _set_pose(self) -> None:
        for camera in self.camera_specs:
            camera_id = self.camera_ids[camera.name]
            x, y, z, pitch, yaw, roll = self._camera_world_pose(camera)
            self._request(f"vset /camera/{camera_id}/location {x} {y} {z}")
            self._request(f"vset /camera/{camera_id}/rotation {pitch} {yaw} {roll}")
        if self.settle_time_s > 0:
            time.sleep(self.settle_time_s)

    def _capture(self) -> dict[str, Any]:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("UnrealCV image capture requires aeroloop[media]") from exc
        images = {}
        for camera in self.camera_specs:
            raw = self._request(f"vget /camera/{self.camera_ids[camera.name]}/lit png")
            bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"failed to decode UnrealCV camera {camera.name!r}")
            images[camera.name] = np.ascontiguousarray(bgr[..., ::-1])
        return images

    def _collision(self) -> tuple[bool, dict[str, Any]]:
        if not self.collision_query:
            return False, {"available": False}
        response = self._request(self.collision_query)
        text = str(response).strip().lower()
        collision = text in {"1", "true", "yes", "collision", "collided"}
        return collision, {"available": True, "raw": str(response)}

    def _observation(self) -> Observation:
        images = self._capture()
        return Observation(
            rgb=images[self.camera_specs[0].name],
            images=images,
            primary_view=self.camera_specs[0].name,
            camera_specs={camera.name: camera for camera in self.camera_specs},
            pose=self.pose,
            relative_state=relative_state(self.pose, self.origin),
            step_index=self.step,
            auxiliary_state={"coordinate_frame": "aeroloop", "native_sdk": "unrealcv"},
        )

    def reset(self, episode: EpisodeSpec) -> Observation:
        self._start()
        self.episode = episode
        self.origin = episode.start_pose
        self.pose = episode.start_pose
        self.step = 0
        self._set_pose()
        return self._observation()

    def execute(self, action: CanonicalAction) -> Transition:
        self.pose = apply_body_action(self.pose, action)
        self._set_pose()
        self.step += 1
        collision, details = self._collision()
        return Transition(self._observation(), collision=collision, info={"collision": details})

    def close(self) -> None:
        if self.client is not None:
            try:
                self.client.disconnect()
            except Exception:
                pass
        if self._owns_process:
            stop_process(self.process)
        self.client = None
        self.process = None
