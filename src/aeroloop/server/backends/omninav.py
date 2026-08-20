from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Sequence

from ..base import ModelBackend, PredictRequest, heading_delta_from_translation


def build_omninav_prompt(instruction: str, history_size: int = 5, horizon: int = 5) -> str:
    """Reproduce the OpenFly OmniNav training prompt exactly."""
    history_prompt = " ".join(f"<input_pos{position}><image>" for position in range(1, history_size + 1))
    return (
        "You are an autonomous aerial navigation robot. Predict the next "
        f"{horizon} three-dimensional body-frame waypoints and whether the mission has ended.\n"
        f"# Historical front-camera observations and UAV positions: {history_prompt}\n"
        "# Current front-camera observation and UAV position: <input_target><image>\n"
        f"# Mission: {instruction.strip()} <|NAV|>\n"
        "Output the waypoint trajectory."
    )


def select_history_indices(available: int, history_size: int = 5) -> list[int]:
    """Match the training loader's uniform history-anchor selection."""
    if available <= 0:
        return [0] * history_size
    if available < history_size:
        return [0] * (history_size - available) + list(range(available))
    if history_size == 1:
        return [available - 1]
    # Equivalent to np.rint(np.linspace(0, available - 1, history_size)).
    return [int(round(index * (available - 1) / (history_size - 1))) for index in range(history_size)]


def history_positions_in_current_body(
    history_states: Sequence[Sequence[float]], current_state: Sequence[float]
) -> list[list[float]]:
    """Express start-relative XYZ history in the current UAV body frame."""
    if len(current_state) < 4:
        raise ValueError("current state must contain [x,y,z,yaw]")
    current_x, current_y, current_z, yaw = map(float, current_state[:4])
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    output = []
    for state in history_states:
        if len(state) < 3:
            raise ValueError("history state must contain XYZ")
        dx = float(state[0]) - current_x
        dy = float(state[1]) - current_y
        dz = float(state[2]) - current_z
        output.append(
            [
                cos_yaw * dx + sin_yaw * dy,
                -sin_yaw * dx + cos_yaw * dy,
                dz,
            ]
        )
    return output


def cumulative_waypoints_to_actions(
    waypoints: Sequence[Sequence[float]], stop_probabilities: Sequence[float]
) -> list[list[float]]:
    """Convert cumulative body-frame XYZ waypoints to canonical action deltas."""
    if len(waypoints) != len(stop_probabilities):
        raise ValueError("waypoint and stop-probability counts must match")
    previous = (0.0, 0.0, 0.0)
    actions = []
    for waypoint, stop in zip(waypoints, stop_probabilities):
        if len(waypoint) < 3:
            raise ValueError("OmniNav waypoint must contain XYZ")
        point = tuple(map(float, waypoint[:3]))
        delta = tuple(point[index] - previous[index] for index in range(3))
        actions.append(
            [
                delta[0],
                delta[1],
                delta[2],
                heading_delta_from_translation(delta[0], delta[1]),
                float(stop),
            ]
        )
        previous = point
    return actions


def _resize_long_edge(image, long_edge: int):
    width, height = image.size
    scale = float(long_edge) / max(width, height)
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(target)


class OmniNavBackend(ModelBackend):
    """Serve the OpenFly-trained OmniNav 3-D waypoint checkpoint."""

    name = "omninav"

    def __init__(
        self,
        repo_root: str,
        ckpt_dir: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        attn_impl: str = "sdpa",
        history_size: int = 5,
        current_long_edge: int = 640,
    ):
        repo = Path(repo_root).expanduser().resolve()
        checkpoint = Path(ckpt_dir).expanduser().resolve()
        transformers_src = repo / "train_code" / "transformers-main" / "src"
        if not transformers_src.is_dir():
            raise FileNotFoundError(f"OmniNav Transformers source not found: {transformers_src}")
        if not (checkpoint / "model.safetensors.index.json").is_file():
            raise FileNotFoundError(f"OmniNav checkpoint is incomplete: {checkpoint}")
        if str(transformers_src) not in sys.path:
            sys.path.insert(0, str(transformers_src))

        import torch
        from PIL import Image
        from transformers import AutoProcessor
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        self.Image = Image
        self.repo_root = repo
        self.ckpt_dir = checkpoint
        self.device = str(device)
        self.dtype_name = str(dtype)
        self.attn_impl = str(attn_impl)
        self.history_size = int(history_size)
        self.current_long_edge = int(current_long_edge)
        if self.history_size != 5:
            raise ValueError("checkpoint-19805 was trained with exactly five history frames")
        if self.current_long_edge <= 0:
            raise ValueError("current_long_edge must be positive")

        torch_dtype = getattr(torch, self.dtype_name)
        checkpoint_path = str(checkpoint)
        self.processor = AutoProcessor.from_pretrained(checkpoint_path, use_fast=False)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            checkpoint_path,
            torch_dtype=torch_dtype,
            attn_implementation=self.attn_impl,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()

        config = self.model.config
        self.horizon = int(config.waypoint_number)
        if int(getattr(config, "waypoint_dim", 0)) != 3:
            raise ValueError(f"expected a 3-D OmniNav checkpoint, got waypoint_dim={config.waypoint_dim}")
        if not bool(getattr(config, "use_arrive_list", False)):
            raise ValueError("OmniNav checkpoint must provide per-waypoint arrive logits")

        self.episode_id: str | None = None
        self.images = []
        self.states: list[tuple[float, float, float, float]] = []

    def health(self):
        return {
            "status": "ok",
            "backend": self.name,
            "checkpoint": str(self.ckpt_dir),
            "device": self.device,
            "dtype": self.dtype_name,
            "attention": self.attn_impl,
            "required_views": ["front"],
            "history_size": self.history_size,
            "native_chunk_size": self.horizon,
            "native_output": "five cumulative 3-D body-frame waypoints plus arrive logits",
            "action_frame": "body",
            "action_semantics": "[dx_body,dy_body,dz_body,d_yaw,stop_probability]",
        }

    def reset(self, episode_id: str, instruction: str, env_name: str):
        self.episode_id = episode_id
        self.images = []
        self.states = []
        return {"episode_id": episode_id}

    def _prepare_history(self, current_image, current_state):
        if self.images:
            indices = select_history_indices(len(self.images), self.history_size)
            history_images = [self.images[index] for index in indices]
            history_states = [self.states[index] for index in indices]
        else:
            history_images = [current_image] * self.history_size
            history_states = [current_state] * self.history_size
        input_points = history_positions_in_current_body(history_states, current_state)
        input_points.append([0.0, 0.0, 0.0])
        return history_images, input_points

    def predict(self, request: PredictRequest):
        if self.episode_id != request.episode_id:
            self.reset(request.episode_id, request.instruction, request.env_name)

        current_state = tuple(map(float, request.state))
        current_image = self.Image.fromarray(request.decode_rgb("front"))
        history_images, input_points = self._prepare_history(current_image, current_state)
        history_images = [_resize_long_edge(image, self.current_long_edge // 4) for image in history_images]
        current_image = _resize_long_edge(current_image, self.current_long_edge)
        images = [*history_images, current_image]

        prompt = build_omninav_prompt(request.instruction, self.history_size, self.horizon)
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        image_token = "<|vision_start|><|image_pad|><|vision_end|>"
        text = text.replace("<image>", image_token)
        inputs = self.processor(text=[text], images=images, padding=True, return_tensors="pt").to(self.device)
        input_waypoints = self.torch.tensor(
            [input_points],
            dtype=self.model.input_wp_encoder.layer1.weight.dtype,
            device=self.device,
        )

        with self.torch.inference_mode():
            waypoint_tensor, arrive_logits = self.model.forward(
                **inputs,
                input_waypoints=input_waypoints,
                action_former=True,
                gt_waypoints=0,
                train=False,
                train_branch=["continue"],
            )
        waypoints = waypoint_tensor[0].float().cpu().tolist()
        stop_probabilities = self.torch.sigmoid(arrive_logits[0]).float().cpu().tolist()
        actions = cumulative_waypoints_to_actions(waypoints, stop_probabilities)

        # Store the unscaled observation so later history preprocessing matches
        # the training-time order: previous frames first, current frame last.
        self.images.append(self.Image.fromarray(request.decode_rgb("front")))
        self.states.append(current_state)
        return {
            "actions": actions,
            "metadata": {
                "model": self.name,
                "checkpoint": str(self.ckpt_dir),
                "action_frame": "body",
                "native_chunk_size": self.horizon,
                "native_waypoints": waypoints,
                "arrive_logits": arrive_logits[0].float().cpu().tolist(),
                "history_observations": len(self.images),
            },
        }
