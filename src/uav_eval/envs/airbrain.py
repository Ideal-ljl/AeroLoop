from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Any

from ..geometry import apply_body_action, relative_state
from ..protocols import EnvironmentAdapter
from ..types import CanonicalAction, EpisodeSpec, Observation, Transition


class AirBrainEnvironment(EnvironmentAdapter):
    """Adapter around AirBrain's AirSim, GS-AirSim, and UnrealCV bridges.

    The bridge renders observations, while UAVEval owns deterministic kinematic
    pose integration and metric state.
    """

    name = "airbrain"

    def __init__(
        self,
        env_name: str,
        *,
        airbrain_root: str,
        color_order: str = "bgr",
        collision_check: bool = True,
        collision_voxel_width: float = 0.2,
        settle_seconds: float = 0.05,
    ):
        self.env_name = env_name
        self.root = Path(airbrain_root).resolve()
        self.scripts_dir = self.root / "scripts"
        self.color_order = color_order.lower()
        if self.color_order not in {"rgb", "bgr"}:
            raise ValueError("color_order must be rgb or bgr")
        self.collision_check = collision_check
        self.collision_voxel_width = float(collision_voxel_width)
        self.settle_seconds = float(settle_seconds)
        self._bridge = None
        self._processor = None
        self._surface_checker = None
        self._collision_checker = None
        self._module = self._load_bridge_module()

    def _load_bridge_module(self):
        path = self.scripts_dir / "env_bridge.py"
        if not path.is_file():
            raise FileNotFoundError(f"AirBrain env_bridge.py not found: {path}")
        scripts = str(self.scripts_dir)
        root = str(self.root)
        if root not in sys.path:
            sys.path.insert(0, root)
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        spec = importlib.util.spec_from_file_location("uav_eval_airbrain_env_bridge", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _create_bridge(self):
        old_cwd = os.getcwd()
        os.chdir(self.root)
        try:
            if "gs" in self.env_name:
                bridge = self._module.GSAirsimBridge(self.env_name)
            elif "airsim" in self.env_name:
                bridge = self._module.AirsimBridge(self.env_name)
            elif "ue" in self.env_name:
                bridge = self._module.UEBridge(ue_ip="127.0.0.1", ue_port="9000", env_name=self.env_name)
            else:
                raise ValueError(f"unsupported AirBrain environment: {self.env_name}")
        finally:
            os.chdir(old_cwd)
        return bridge

    def _load_pos_ratio(self) -> float:
        if "gs" not in self.env_name:
            return 1.0
        import yaml

        candidates = [
            self.root / "configs" / f"{self.env_name}.yaml",
            Path(os.environ.get("OPENFLY_ENV_CONFIGS_DIR", "")) / f"{self.env_name}.yaml",
        ]
        for path in candidates:
            if path.is_file():
                raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                return float(raw.get("traj_map", {}).get("pcd_scale_ratio", 5.0))
        raise FileNotFoundError(f"missing pcd_scale_ratio config for {self.env_name}; checked {candidates}")

    def _point_cloud_dir(self) -> Path:
        for rel in ("dataset/dense_ins_data", "dataset/dense_ins_data0"):
            path = self.root / rel / self.env_name
            if path.is_dir():
                return path
        raise FileNotFoundError(f"point cloud folder not found for {self.env_name}")

    def _ensure_runtime(self) -> None:
        if self._bridge is not None:
            return
        self._bridge = self._create_bridge()
        ply = self._point_cloud_dir()
        if "gs" in self.env_name:
            self._processor = self._module.TrajectoryProcessor(
                str(ply),
                img_width=self._bridge.camera_width,
                img_height=self._bridge.camera_height,
                fov=self._bridge.camera_fov,
            )
        else:
            self._processor = self._module.TrajectoryProcessor(str(ply))
        self._surface_checker = self._module.SurfaceDistanceChecker.from_trajectory_processor(self._processor)
        if self.collision_check:
            self._collision_checker = self._module.PointCloudCollisionChecker.from_point_clouds(
                self._processor.buildings,
                voxel_width=self.collision_voxel_width,
                dilate_radius=0.0,
            )
        self.pos_ratio = self._load_pos_ratio()

    def _set_pose(self) -> None:
        pitch = float(self.episode.metadata.get("pitch", 0.0))
        self._bridge.set_camera_pose(
            self.pose.x / self.pos_ratio,
            self.pose.y / self.pos_ratio,
            self.pose.z / self.pos_ratio,
            pitch,
            self.pose.yaw * 180.0 / 3.141592653589793,
            0.0,
        )
        if self.settle_seconds > 0:
            time.sleep(self.settle_seconds)

    def _rgb(self):
        image = self._bridge.get_camera_data()
        if self.color_order == "bgr":
            if getattr(image, "ndim", 0) == 3 and image.shape[-1] == 3:
                image = image[..., ::-1].copy()
        return image

    def _observation(self, info: dict[str, Any] | None = None) -> Observation:
        return Observation(
            rgb=self._rgb(),
            pose=self.pose,
            relative_state=relative_state(self.pose, self.origin),
            step_index=self.step,
            info=info or {},
        )

    def reset(self, episode: EpisodeSpec) -> Observation:
        if episode.env_name != self.env_name:
            raise ValueError(f"environment instance is {self.env_name}, episode requests {episode.env_name}")
        self._ensure_runtime()
        self.episode = episode
        self.origin = episode.start_pose
        self.pose = episode.start_pose
        self.step = 0
        self._set_pose()
        return self._observation()

    def execute(self, action: CanonicalAction) -> Transition:
        before = self.pose
        self.pose = apply_body_action(before, action)
        self.step += 1
        self._set_pose()
        collision = False
        if self._collision_checker is not None:
            collision = bool(self._collision_checker.is_collision(before.xyz(), self.pose.xyz()))
        surface_distance = float("inf")
        if self.episode.target_id is not None:
            surface_distance = self._surface_checker.distance_to_surface(self.pose.xyz(), self.episode.target_id)
        info = {"surface_distance": surface_distance}
        return Transition(self._observation(info), collision=collision, info=info)

    def close(self) -> None:
        if self._bridge is None:
            return
        if hasattr(self._bridge, "stop"):
            self._bridge.stop()
        elif getattr(self._bridge, "process", None) is not None:
            process = self._bridge.process
            if process.poll() is None:
                process.terminate()
        self._bridge = None
