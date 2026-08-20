from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path
from typing import Sequence

from ..base import ModelBackend, PredictRequest, heading_delta_from_translation


def pi0_actions_to_canonical(actions: Sequence[Sequence[float]]) -> list[list[float]]:
    """Convert OpenFly PI0 [dx_body,dy_body,dz_body,stop] chunks to AeroLoop actions.

    The legacy OpenFly evaluator updates yaw by ``atan2(dy_body, dx_body)`` for
    four-dimensional actions. Reproducing that behavior is required because the
    PI0 checkpoint has no explicit yaw output channel.
    """

    output = []
    for index, action in enumerate(actions):
        if len(action) != 4:
            raise ValueError(f"PI0 action {index} must contain [dx,dy,dz,stop], got {action}")
        dx, dy, dz, stop = map(float, action)
        if not all(math.isfinite(value) for value in (dx, dy, dz, stop)):
            raise ValueError(f"PI0 action {index} contains NaN or infinity")
        output.append(
            [
                dx,
                dy,
                dz,
                heading_delta_from_translation(dx, dy),
                min(1.0, max(0.0, stop)),
            ]
        )
    return output


class PI0Backend(ModelBackend):
    """Serve the OpenFly-finetuned LeRobot PI0 checkpoint."""

    name = "pi0"

    def __init__(
        self,
        repo_root: str,
        ckpt_dir: str,
        tokenizer_dir: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        seed: int = 2026,
        inference_steps: int | None = None,
    ):
        repo = Path(repo_root).expanduser().resolve()
        checkpoint = Path(ckpt_dir).expanduser().resolve()
        tokenizer = Path(tokenizer_dir).expanduser().resolve()
        for path, description in (
            (repo / "src" / "lerobot", "LeRobot source"),
            (checkpoint / "model.safetensors", "PI0 model weights"),
            (checkpoint / "config.json", "PI0 config"),
            (checkpoint / "policy_preprocessor.json", "PI0 preprocessor"),
            (checkpoint / "policy_postprocessor.json", "PI0 postprocessor"),
            (tokenizer / "tokenizer.model", "PaliGemma tokenizer"),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{description} not found: {path}")

        source = str(repo / "src")
        if source not in sys.path:
            sys.path.insert(0, source)

        import torch
        import transformers
        from safetensors import safe_open

        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi0 import PI0Policy

        self.torch = torch
        self.repo_root = repo
        self.ckpt_dir = checkpoint
        self.tokenizer_dir = tokenizer
        self.device = str(device)
        self.dtype_name = str(dtype)
        self.seed = int(seed)
        self.transformers_version = transformers.__version__

        config = PreTrainedConfig.from_pretrained(checkpoint)
        config.device = self.device
        config.dtype = self.dtype_name
        config.gradient_checkpointing = False
        config.compile_model = False
        config.tokenizer_name = str(tokenizer)
        if inference_steps is not None:
            config.num_inference_steps = int(inference_steps)
        self.config = config

        self.model = PI0Policy.from_pretrained(
            checkpoint,
            config=config,
            strict=True,
            local_files_only=True,
        )
        self.model.eval()

        # PI0Policy.from_pretrained logs and returns an initialized model when a
        # load fails. Verify one learned tensor explicitly so such a failure can
        # never become a silent benchmark run with random weights.
        verification_key = "model.action_out_proj.weight"
        with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as handle:
            saved = handle.get_tensor(verification_key)
        loaded = self.model.state_dict()[verification_key].detach().cpu()
        if not torch.equal(saved, loaded):
            raise RuntimeError(f"checkpoint verification failed for {verification_key}")

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            config,
            str(checkpoint),
            preprocessor_overrides={
                "tokenizer_processor": {"tokenizer_name": str(tokenizer)},
                "device_processor": {"device": self.device},
            },
        )
        self.episode_id: str | None = None
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(self.seed)

    def health(self):
        return {
            "status": "ok",
            "backend": self.name,
            "checkpoint": str(self.ckpt_dir),
            "tokenizer": str(self.tokenizer_dir),
            "device": self.device,
            "dtype": self.dtype_name,
            "transformers": self.transformers_version,
            "required_views": ["front"],
            "state": "start-frame-relative [x,y,z,yaw]",
            "native_chunk_size": int(self.config.chunk_size),
            "observation_steps": int(self.config.n_obs_steps),
            "history_order": "oldest_to_current",
            "inference_steps": int(self.config.num_inference_steps),
            "native_output": "[dx_body,dy_body,dz_body,stop]",
            "action_semantics": "[dx_body,dy_body,dz_body,atan2(dy,dx),clamped_stop]",
            "checkpoint_verified": True,
        }

    def reset(self, episode_id: str, instruction: str, env_name: str):
        self.episode_id = episode_id
        self.model.reset()
        digest = hashlib.sha256(episode_id.encode("utf-8")).digest()
        episode_seed = self.seed + int.from_bytes(digest[:4], "little")
        self.generator.manual_seed(episode_seed)
        return {"episode_id": episode_id, "seed": episode_seed}

    def predict(self, request: PredictRequest):
        if self.episode_id != request.episode_id:
            self.reset(request.episode_id, request.instruction, request.env_name)
        if not request.instruction.strip():
            raise ValueError("PI0 requires a non-empty OpenFly instruction")

        images = request.decode_rgb_history("front")
        observation_steps = int(self.config.n_obs_steps)
        images = images[-observation_steps:]
        while len(images) < observation_steps:
            images.insert(0, images[0])
        image_tensor = self.torch.stack(
            [
                self.torch.from_numpy(image.copy())
                .permute(2, 0, 1)
                .to(dtype=self.torch.float32)
                .div_(255.0)
                for image in images
            ]
        ).unsqueeze(0)
        state = self.torch.tensor(request.state, dtype=self.torch.float32).unsqueeze(0)
        batch = self.preprocessor(
            {
                "observation.images.image": image_tensor,
                "observation.state": state,
                "task": request.instruction,
            }
        )
        noise = self.torch.randn(
            1,
            self.config.chunk_size,
            self.config.max_action_dim,
            generator=self.generator,
            device=self.device,
            dtype=self.torch.float32,
        )
        with self.torch.inference_mode():
            normalized = self.model.predict_action_chunk(
                batch,
                noise=noise,
                num_steps=self.config.num_inference_steps,
            )
            actions = self.postprocessor(normalized)
        native_actions = actions[0].float().cpu().tolist()
        canonical_actions = pi0_actions_to_canonical(native_actions)
        return {
            "actions": canonical_actions,
            "metadata": {
                "model": self.name,
                "checkpoint": str(self.ckpt_dir),
                "action_frame": "body",
                "native_chunk_size": len(native_actions),
                "observation_steps": observation_steps,
                "native_actions": native_actions,
                "step": request.step,
            },
        }
