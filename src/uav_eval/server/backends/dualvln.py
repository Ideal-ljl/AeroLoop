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
        self.predict_steps = int(predict_steps)
        self.inference_steps = int(inference_steps)
        self.sample_trajectories = int(sample_trajectories)
        self.stop_speed_threshold = float(stop_speed_threshold)
        self.seed = int(seed)
        stage2 = self.ckpt_dir / "stage2"
        stage1 = self.ckpt_dir / "stage1"
        depth = self.ckpt_dir / "depth_anything_v2_metric_hypersim_vits.pth"
        for path in (self.repo_root, stage1, stage2, depth):
            if not path.exists():
                raise FileNotFoundError(path)
        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))

        import torch
        from transformers import AutoProcessor, AutoTokenizer
        from internnav.model.basemodel.internvla_n1 import internvla_n1_arch
        from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM

        # The upstream module uses a cwd-relative constant during construction.
        internvla_n1_arch.MODEL_PATH_TO = str(self.ckpt_dir)
        self.torch = torch
        torch_dtype = getattr(torch, dtype)
        self.model = InternVLAN1ForCausalLM.from_pretrained(
            stage2,
            torch_dtype=torch_dtype,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).to(device)
        self.model.eval()
        # The stage-2 export has the tokenizer but omits preprocessor_config.json.
        # Stage 1 and stage 2 use the same Qwen vision frontend, so compose them.
        self.processor = AutoProcessor.from_pretrained(stage1)
        self.processor.tokenizer = AutoTokenizer.from_pretrained(stage2, use_fast=True)
        self.processor.tokenizer.padding_side = "left"
        self.episode_id = None

    def health(self):
        return {
            "status": "ok",
            "backend": self.name,
            "checkpoint": str(self.ckpt_dir),
            "device": self.device,
            "dtype": self.dtype_name,
            "native_chunk_size": self.predict_steps,
            "sample_trajectories": self.sample_trajectories,
            "system1": getattr(self.model.config, "system1", None),
        }

    def reset(self, episode_id: str, instruction: str, env_name: str):
        self.episode_id = episode_id
        self.model.rope_deltas = None
        return {"episode_id": episode_id}

    def predict(self, request: PredictRequest):
        import numpy as np
        from PIL import Image

        if self.episode_id != request.episode_id:
            self.reset(request.episode_id, request.instruction, request.env_name)
        self.torch.manual_seed(self.seed + request.step)
        rgb = request.decode_rgb()
        image = Image.fromarray(rgb)
        prompt = (
            "You are an autonomous navigation assistant. Follow the instruction using the current UAV observation; "
            "the trajectory will be produced by the System1 head.\n"
            f"Instruction: {request.instruction}\nObservation:"
        )
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)
        image_grid = inputs.image_grid_thw
        with self.torch.inference_mode():
            traj_latents = self.model.generate_latents(inputs.input_ids, inputs.pixel_values, image_grid)
            resized = np.asarray(image.resize((224, 224)), dtype=np.float32) / 255.0
            image_pair = self.torch.from_numpy(np.stack([resized, resized], axis=0)).unsqueeze(0).to(
                self.device, dtype=self.model.dtype
            )
            sampled = self.model.generate_traj(
                traj_latents=traj_latents,
                images_dp=image_pair,
                predict_step_nums=self.predict_steps,
                num_inference_steps=self.inference_steps,
                num_sample_trajs=self.sample_trajectories,
            )
        samples = sampled.detach().float().cpu().numpy().reshape(-1, self.predict_steps, 3)
        trajectory = samples.mean(axis=0)
        deltas = np.diff(np.concatenate([np.zeros((1, 3), dtype=np.float32), trajectory], axis=0), axis=0)
        speeds = np.linalg.norm(deltas, axis=1)
        stops = np.zeros(self.predict_steps, dtype=np.float32)
        if float(speeds[-3:].mean()) < self.stop_speed_threshold:
            stops[-1] = 1.0
        actions = [
            [float(delta[0]), float(delta[1]), float(delta[2]), 0.0, float(stops[index])]
            for index, delta in enumerate(deltas)
        ]
        return {
            "actions": actions,
            "metadata": {
                "model": self.name,
                "native_chunk_size": self.predict_steps,
                "trajectory_samples": int(samples.shape[0]),
                "trajectory_is_cumulative": True,
            },
        }
