from __future__ import annotations

import sys
from pathlib import Path

from ..base import ModelBackend, PredictRequest


def build_aerialvla_prompt(instruction: str) -> str:
    """Build the exact prompt used by AerialVLA training."""
    return f"<image>\n{instruction.strip()}\nAction: "


class AerialVLABackend(ModelBackend):
    name = "aerialvla"

    def __init__(self, repo_root: str, ckpt_dir: str, device: str = "cuda", dtype: str = "bfloat16"):
        self.repo_root = Path(repo_root).resolve()
        self.ckpt_dir = Path(ckpt_dir).resolve()
        self.device = device
        self.dtype_name = dtype
        for path in (self.repo_root, self.ckpt_dir / "openvla-7b", self.ckpt_dir / "lora"):
            if not path.exists():
                raise FileNotFoundError(path)
        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))

        import torch
        import transformers
        from peft import PeftModel

        # The shared uavflow environment contains a flash-attn wheel built
        # against a different PyTorch ABI.  AerialVLA uses eager attention,
        # so prevent Transformers from importing that optional broken wheel.
        transformers.utils.is_flash_attn_2_available = lambda: False
        transformers.utils.import_utils.is_flash_attn_2_available = lambda: False
        from transformers import AutoImageProcessor, AutoModelForVision2Seq, AutoTokenizer

        self.torch = torch
        self.parse_action = __import__(
            "src.aerialvla_action_codec", fromlist=["parse_aerialvla_action_text"]
        ).parse_aerialvla_action_text
        base = self.ckpt_dir / "openvla-7b"
        lora = self.ckpt_dir / "lora"
        # Match the native wrapper: token ids come from the complete base
        # checkpoint's fast tokenizer.  The LoRA export has an incompatible
        # tokenizer.json and no tokenizer.model, so it is metadata only.
        self.tokenizer = AutoTokenizer.from_pretrained(
            base, trust_remote_code=True, use_fast=True
        )
        self.image_processor = AutoImageProcessor.from_pretrained(base, trust_remote_code=True)
        torch_dtype = getattr(torch, dtype)
        model = AutoModelForVision2Seq.from_pretrained(
            base,
            torch_dtype=torch_dtype,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model.resize_token_embeddings(len(self.tokenizer))
        self.model = PeftModel.from_pretrained(model, lora).to(device)
        self.model.eval()
        self.episode_id = None

    def health(self):
        return {
            "status": "ok",
            "backend": self.name,
            "checkpoint": str(self.ckpt_dir),
            "device": self.device,
            "dtype": self.dtype_name,
            "native_chunk_size": 1,
            "action_frame": "body",
            "action_semantics": "[forward,0,down,d_yaw,LAND]",
            "action_codec": "src.aerialvla_action_codec.parse_aerialvla_action_text",
        }

    def reset(self, episode_id: str, instruction: str, env_name: str):
        self.episode_id = episode_id
        return {"episode_id": episode_id}

    def predict(self, request: PredictRequest):
        from PIL import Image

        if self.episode_id != request.episode_id:
            self.reset(request.episode_id, request.instruction, request.env_name)
        rgb = request.decode_rgb()
        image = Image.fromarray(rgb)
        prompt = build_aerialvla_prompt(request.instruction)
        text = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        pixels = self.image_processor(images=image, return_tensors="pt")["pixel_values"].to(
            self.device, dtype=self.model.dtype
        )
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=text["input_ids"],
                attention_mask=text.get("attention_mask"),
                pixel_values=pixels,
                max_new_tokens=20,
                do_sample=False,
                use_cache=True,
                eos_token_id=[self.tokenizer.eos_token_id],
            )
        decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=False)
        forward, down, d_yaw, stop = self.parse_action(decoded)
        action = [float(forward), 0.0, float(down), float(d_yaw), float(stop)]
        return {
            "actions": [action],
            "metadata": {
                "model": self.name,
                "action_frame": "body",
                "raw_output": decoded,
                "native_action": [float(forward), float(down), float(d_yaw), bool(stop)],
            },
        }
