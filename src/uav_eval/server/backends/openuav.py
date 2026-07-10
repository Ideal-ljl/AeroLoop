from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Mapping

from ..base import ModelBackend, PredictRequest


_REQUIRED_NAVIGATION_WEIGHTS = (
    "embed_tokens",
    "waypoint_emb",
    "waypoints_fc",
    "waypoints_output",
    "history_preprocessor",
)


def missing_navigation_weights(keys: Iterable[str]) -> list[str]:
    """Return OpenUAV modules that are absent from an exported checkpoint."""
    names = tuple(keys)
    return [module for module in _REQUIRED_NAVIGATION_WEIGHTS if not any(module in key for key in names)]


def normalize_checkpoint_state(state: Mapping[str, object], model_keys: Iterable[str]) -> dict[str, object]:
    """Map PEFT/Distributed wrapper keys back to the unwrapped OpenUAV model."""
    expected = set(model_keys)
    normalized = {}
    prefixes = ("module.", "base_model.model.")
    for original, value in state.items():
        candidates = [original]
        candidate = original
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if candidate.startswith(prefix):
                    candidate = candidate[len(prefix) :]
                    candidates.append(candidate)
                    changed = True
                    break
        match = next((key for key in candidates if key in expected), None)
        if match is not None:
            normalized[match] = value
    return normalized


class OpenUAVBackend(ModelBackend):
    name = "openuav"

    def __init__(
        self,
        repo_root: str,
        ckpt_dir: str,
        device: str = "cuda",
        stop_norm_threshold: float = 0.5,
        history_size: int = 4,
    ):
        self.repo_root = Path(repo_root).resolve()
        self.ckpt_dir = Path(ckpt_dir).resolve()
        self.device = device
        self.stop_norm_threshold = float(stop_norm_threshold)
        self.history_size = int(history_size)
        model_root = self.repo_root / "Model" / "LLaMA-UAV"
        for path in (model_root, self.ckpt_dir / "final", self.ckpt_dir / "model_zoo" / "vicuna-7b-v1.5"):
            if not path.exists():
                raise FileNotFoundError(path)
        for path in (self.repo_root, model_root):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))

        import torch
        self.torch = torch
        final = self.ckpt_dir / "final"
        base = self.ckpt_dir / "model_zoo" / "vicuna-7b-v1.5"
        self.has_lora = (final / "adapter_config.json").is_file() and (
            (final / "adapter_model.safetensors").is_file() or (final / "adapter_model.bin").is_file()
        )
        checkpoint_states = self._read_checkpoint_states(final)
        missing = missing_navigation_weights(key for state in checkpoint_states for key in state)
        if missing:
            raise RuntimeError(
                f"incomplete OpenUAV checkpoint {final}: missing trained weights for "
                f"{', '.join(missing)}; refusing to serve randomly initialized navigation heads"
            )

        from llamavid import conversation as conversation_lib
        from llamavid.constants import DEFAULT_HISTORY_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_WP_TOKEN
        from llamavid.train.train_uav.train_uav_notice import preprocess

        self.preprocess = preprocess
        self.DEFAULT_HISTORY_TOKEN = DEFAULT_HISTORY_TOKEN
        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.DEFAULT_WP_TOKEN = DEFAULT_WP_TOKEN
        if "imgsp_uav" in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates["imgsp_uav"]

        self.tokenizer, self.model, self.image_processor = self._load_base_model(base, final)
        loaded_keys = set()
        model_keys = self.model.state_dict().keys()
        for state in checkpoint_states:
            normalized = normalize_checkpoint_state(state, model_keys)
            self.model.load_state_dict(normalized, strict=False)
            loaded_keys.update(normalized)
        missing_after_load = missing_navigation_weights(loaded_keys)
        if missing_after_load:
            raise RuntimeError(
                "OpenUAV checkpoint keys could not be mapped to the model: " + ", ".join(missing_after_load)
            )
        self.loaded_checkpoint_keys = len(loaded_keys)

        if self.has_lora:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, final)
        self.model.to(device)
        self.model.eval()
        self.episode_id = None
        self.state_history = []

    def _torch_load(self, path: Path):
        try:
            return self.torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # torch<2.0
            return self.torch.load(path, map_location="cpu")

    def _read_checkpoint_states(self, final: Path) -> list[Mapping[str, object]]:
        paths = [final / "mm_projector.bin", final / "non_lora_trainables.bin"]
        states = []
        for path in paths:
            if path.is_file():
                state = self._torch_load(path)
                if not isinstance(state, Mapping):
                    raise TypeError(f"checkpoint must contain a state dict: {path}")
                states.append(state)
        if not states:
            raise FileNotFoundError(f"no OpenUAV adapter weights found under {final}")
        return states

    def _load_base_model(self, base: Path, final: Path):
        from transformers import AutoConfig, AutoTokenizer

        from llamavid.model.language_model.llava_llama_uav import LlavaConfig, LlavaLlamaAttForCausalLM

        config = LlavaConfig.from_pretrained(final)
        base_config = AutoConfig.from_pretrained(base)
        config.vocab_size = base_config.vocab_size

        local_vision_tower = self.ckpt_dir / "model_zoo" / "LAVIS" / "eva_vit_g.pth"
        local_processor = self.repo_root / "Model" / "LLaMA-UAV" / "llamavid" / "processor" / "clip-patch14-224"
        if local_vision_tower.is_file():
            config.mm_vision_tower = str(local_vision_tower)
        if local_processor.is_dir():
            config.image_processor = str(local_processor)

        tokenizer = AutoTokenizer.from_pretrained(base, use_fast=False)
        model = LlavaLlamaAttForCausalLM.from_pretrained(
            base,
            config=config,
            low_cpu_mem_usage=True,
            torch_dtype=self.torch.bfloat16,
        )
        self.tokenizer = tokenizer
        self.model = model
        self._add_navigation_tokens()

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=self.device, dtype=self.torch.bfloat16)
        model.config.model_path = str(final)
        model.get_model().initialize_attention_modules(model.config, for_eval=True)
        return tokenizer, model, vision_tower.image_processor

    def _add_navigation_tokens(self) -> None:
        added = self.tokenizer.add_tokens(["<wp>", "<his>"], special_tokens=True)
        old_count = self.model.get_input_embeddings().weight.shape[0]
        self.model.resize_token_embeddings(len(self.tokenizer))
        if added > 0 and len(self.tokenizer) > old_count:
            input_embeddings = self.model.get_input_embeddings().weight.data
            output_embeddings = self.model.get_output_embeddings().weight.data
            input_embeddings[-added:] = input_embeddings[:-added].mean(dim=0, keepdim=True)
            output_embeddings[-added:] = output_embeddings[:-added].mean(dim=0, keepdim=True)
        token_ids = {
            "<wp>": self.tokenizer.convert_tokens_to_ids("<wp>"),
            "<his>": self.tokenizer.convert_tokens_to_ids("<his>"),
            ",": self.tokenizer.encode(",", add_special_tokens=False)[-1],
            ";": self.tokenizer.encode(";", add_special_tokens=False)[-1],
        }
        self.model.get_special_token_id(token_ids)

    @staticmethod
    def _format_vector(values) -> str:
        return ",".join(str(round(float(value), 3)) for value in values[:3])

    def health(self):
        warning = None if self.has_lora else "checkpoint has no LoRA adapter; serving projector/waypoint weights only"
        return {
            "status": "ok",
            "backend": self.name,
            "checkpoint": str(self.ckpt_dir),
            "device": self.device,
            "native_chunk_size": 1,
            "lora_loaded": self.has_lora,
            "checkpoint_keys_loaded": self.loaded_checkpoint_keys,
            "warning": warning,
        }

    def reset(self, episode_id: str, instruction: str, env_name: str):
        self.episode_id = episode_id
        self.state_history = []
        return {"episode_id": episode_id}

    def _history_waypoint(self, current_state):
        torch = self.torch
        positions = self.state_history[-self.history_size :] + [tuple(current_state[:3])]
        return torch.tensor(positions, dtype=torch.float32).reshape(-1)

    def predict(self, request: PredictRequest):
        from PIL import Image

        if self.episode_id != request.episode_id:
            self.reset(request.episode_id, request.instruction, request.env_name)
        rgb = request.decode_rgb()
        image_tensor = self.image_processor.preprocess(Image.fromarray(rgb), return_tensors="pt")["pixel_values"][0]
        history = self._history_waypoint(request.state)
        previous = [0.0, 0.0, 0.0]
        if len(self.state_history) >= 1:
            previous = [request.state[i] - self.state_history[-1][i] for i in range(3)]
        instruction = request.instruction.strip()
        sources = [[
            {
                "from": "human",
                "value": (
                    "Stage:cruise\n\n"
                    f"Previous displacement:{self._format_vector(previous)}\n\n"
                    f"Current position:{self._format_vector(request.state)}\n\n"
                    f"History images:{self.DEFAULT_HISTORY_TOKEN}\n\n"
                    f"History waypoints:{self.DEFAULT_WP_TOKEN}\n\n"
                    f"Current image:{self.DEFAULT_IMAGE_TOKEN}\n\n"
                    f"Instruction:{instruction}"
                ),
                "prompt": instruction,
            },
            {"from": "gpt", "value": ""},
        ]]
        processed = self.preprocess(sources, self.tokenizer, has_image=True, refine_prompt=False)
        input_ids = processed["input_ids"].to(self.device)
        labels = processed["labels"].to(self.device)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        prompt_values = processed.get("prompt") or [instruction]
        prompts = [prompt_values]
        with self.torch.inference_mode():
            waypoint = self.model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                images=[image_tensor.to(self.device, dtype=self.model.dtype)],
                prompts=prompts,
                historys=[history.to(self.device, dtype=self.model.dtype)],
                return_waypoints=True,
                use_cache=False,
            )
        wp = waypoint.detach().float().cpu().reshape(-1)
        unit = wp[:3]
        norm = float(wp[3])
        length = float(self.torch.linalg.vector_norm(unit))
        if length > 1e-6:
            unit = unit / length
        delta = unit * norm
        stop = 1.0 if norm < self.stop_norm_threshold else 0.0
        self.state_history.append(tuple(request.state[:3]))
        return {
            "actions": [[float(delta[0]), float(delta[1]), float(delta[2]), 0.0, stop]],
            "metadata": {
                "model": self.name,
                "native_waypoint": [float(x) for x in wp[:4]],
                "lora_loaded": self.has_lora,
            },
        }
