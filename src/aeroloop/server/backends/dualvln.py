from __future__ import annotations

import sys
from pathlib import Path

from ..base import ModelBackend, PredictRequest


class DualVLNBackend(ModelBackend):
    name = "dualvln"

    def __init__(
        self,
        repo_root: str,
        ckpt_dir: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        predict_steps: int = 32,
        inference_steps: int = 10,
        sample_trajectories: int = 32,
        stop_speed_threshold: float = 0.05,
        seed: int = 2026,
    ):
        self.repo_root = Path(repo_root).resolve()
        self.ckpt_dir = Path(ckpt_dir).resolve()
        self.device = device
        self.dtype_name = dtype
        self.action_horizon = int(predict_steps)
        self.system2_refresh_steps = int(sample_trajectories)
        self.stop_threshold = float(stop_speed_threshold)
        self.seed = int(seed)
        stage2 = self.ckpt_dir / "stage2"
        stage1 = self.ckpt_dir / "stage1"
        for path in (self.repo_root, stage1, stage2):
            if not path.exists():
                raise FileNotFoundError(path)
        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))

        import numpy as np
        import torch
        from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ModelConfig
        from internnav.model.basemodel.internvla_n1.internvla_n1_policy import InternVLAN1Net

        self.np = np
        self.torch = torch
        config = InternVLAN1ModelConfig(
            model_cfg={
                "model": {
                    "policy_name": "InternVLAN1_Policy",
                    "state_encoder": None,
                    "model_path": str(stage2),
                    "processor_path": str(stage1),
                    "device": device,
                    "attn_implementation": "sdpa",
                    "num_frames": 1,
                    "num_history": 8,
                    "num_future_steps": 32,
                    "continuous_traj": True,
                    "resize_w": 384,
                    "resize_h": 384,
                    "system1_image_size": 224,
                    "uav_action_horizon": self.action_horizon,
                }
            }
        )
        self.policy = InternVLAN1Net(config)
        self.episode_id = None
        self.latent = None
        self.prompt = None
        self.anchor_steps = 0

    def health(self):
        checkpoint_config = getattr(getattr(self.policy, "model", None), "config", None)
        return {
            "status": "ok",
            "backend": self.name,
            "checkpoint": str(self.ckpt_dir),
            "device": self.device,
            "dtype": self.dtype_name,
            "native_chunk_size": self.action_horizon,
            "system2_refresh_steps": self.system2_refresh_steps,
            "trajectory_representation": getattr(self.policy, "trajectory_representation", None),
            "system2_inference_mode": getattr(self.policy, "system2_inference_mode", None),
            "openfly_conditioning": getattr(checkpoint_config, "openfly_conditioning", None),
            "openfly_action_scale": getattr(checkpoint_config, "openfly_action_scale", None),
        }

    def reset(self, episode_id: str, instruction: str, env_name: str):
        self.episode_id = episode_id
        self.policy.reset()
        self.latent = None
        self.prompt = None
        self.anchor_steps = 0
        return {"episode_id": episode_id}

    def predict(self, request: PredictRequest):
        from PIL import Image

        if self.episode_id != request.episode_id:
            self.reset(request.episode_id, request.instruction, request.env_name)
        self.torch.manual_seed(self.seed + request.step)
        rgb = request.decode_rgb()
        image = Image.fromarray(rgb)
        prompt = (
            request.instruction or ""
        )
        refresh_system2 = (
            self.latent is None
            or self.prompt != prompt
            or self.anchor_steps >= self.system2_refresh_steps
        )
        with self.torch.inference_mode():
            if refresh_system2:
                self.policy.reset()
                s2_output = self.policy.s2_step(rgb, None, None, prompt, None)
                if s2_output.output_latent is None:
                    raise RuntimeError("DualVLN System 2 did not produce a latent plan.")
                self.latent = s2_output.output_latent
                self.prompt = prompt
                self.anchor_steps = 0
            output = self.policy.s1_step_latent(rgb, None, self.latent, output_mode="openfly")
        raw_actions = self.np.asarray(output.action_chunk, dtype=self.np.float32).copy()
        if raw_actions.ndim != 2 or raw_actions.shape[1] != 4 or not self.np.isfinite(raw_actions).all():
            raise RuntimeError(f"Invalid DualVLN System 1 action chunk: {raw_actions.shape}")
        self.anchor_steps += len(raw_actions)
        actions = raw_actions.tolist()
        return {
            "actions": actions,
            "metadata": {
                "model": self.name,
                "native_chunk_size": self.action_horizon,
                "openfly_action_contract": "dxyz_stop",
                "system2_refreshed": refresh_system2,
                "anchor_steps": self.anchor_steps,
            },
        }
