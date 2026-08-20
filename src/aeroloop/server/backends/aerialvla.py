from __future__ import annotations

import re
import sys
from pathlib import Path

from ..base import ModelBackend, PredictRequest


def build_aerialvla_prompt(instruction: str) -> str:
    """Build the exact prompt used by AerialVLA training."""
    instruction = instruction.strip()
    if not instruction.startswith("Instruction:"):
        instruction = f"Instruction:{instruction}"
    if not instruction.endswith(";"):
        instruction = f"{instruction};"
    return f"<image>\n{instruction}\nAction: "


def original_openvla_action_token_ids(config) -> tuple[int, ...]:
    """Return the base OpenVLA token interval reserved for robot actions."""
    text_config = getattr(config, "text_config", None)
    padded_vocab_size = int(getattr(text_config, "vocab_size", 0))
    pad_to_multiple_of = int(getattr(config, "pad_to_multiple_of", 0))
    action_bins = int(getattr(config, "n_action_bins", 0))
    effective_vocab_size = padded_vocab_size - pad_to_multiple_of
    first_action_id = effective_vocab_size - action_bins
    if action_bins <= 0 or first_action_id < 0:
        raise ValueError(
            "Invalid OpenVLA action-token configuration: "
            f"vocab={padded_vocab_size} padding={pad_to_multiple_of} bins={action_bins}"
        )
    return tuple(range(first_action_id, effective_vocab_size))


def validate_aerialvla_action_text(text: str, num_bins: int = 99) -> tuple[int, int, int, bool]:
    """Require exactly three in-range text bins and an optional ``<LAND>``."""
    action_text = text.rsplit("Action:", 1)[-1]
    action_text = action_text.split("</s>", 1)[0].strip()
    match = re.fullmatch(r"(\d+)\s+(\d+)\s+(\d+)(?:\s+(<LAND>))?", action_text)
    if match is None:
        raise ValueError(f"Malformed AerialVLA action text: {action_text!r}")
    bins = tuple(int(value) for value in match.group(1, 2, 3))
    if not all(0 <= value < int(num_bins) for value in bins):
        raise ValueError(f"AerialVLA action bins outside [0, {int(num_bins) - 1}]: {bins}")
    return bins[0], bins[1], bins[2], match.group(4) is not None


def greedy_generate_no_cache(
    model,
    *,
    input_ids,
    attention_mask,
    pixel_values,
    max_new_tokens: int,
    eos_token_id: int,
    forbidden_token_ids: tuple[int, ...],
    tokenizer,
    num_bins: int,
):
    """Run deterministic full-prefix decoding with the trained text grammar."""
    import torch

    if input_ids.shape[0] != 1:
        raise ValueError("AerialVLA constrained generation currently requires batch size one")
    separator_and_zero = tokenizer.encode("0", add_special_tokens=False)
    if len(separator_and_zero) != 2:
        raise ValueError(f"Unexpected tokenizer representation for action bins: {separator_and_zero}")
    separator_id = int(separator_and_zero[0])
    digit_ids = []
    for digit in range(10):
        encoded = tokenizer.encode(str(digit), add_special_tokens=False)
        if len(encoded) != 2 or int(encoded[0]) != separator_id:
            raise ValueError(f"Unexpected tokenizer representation for digit {digit}: {encoded}")
        digit_ids.append(int(encoded[1]))
    land_ids = [int(value) for value in tokenizer.encode(" <LAND>", add_special_tokens=False)]
    if not land_ids or land_ids[0] != separator_id:
        raise ValueError(f"Unexpected tokenizer representation for <LAND>: {land_ids}")
    land_tail = land_ids[1:]

    generated = input_ids
    mask = attention_mask
    finished = torch.zeros(generated.shape[0], dtype=torch.bool, device=generated.device)
    forbidden = set(map(int, forbidden_token_ids))
    state = "leading_separator"
    field_index = 0
    first_digit = None
    land_index = 0
    for _ in range(int(max_new_tokens)):
        outputs = model(
            input_ids=generated,
            attention_mask=mask,
            pixel_values=pixel_values,
            use_cache=False,
            return_dict=True,
        )
        logits = outputs.logits[:, -1, :]
        if state == "leading_separator":
            allowed = [separator_id]
        elif state == "first_digit":
            allowed = digit_ids
        elif state == "number_tail":
            allowed = []
            if first_digit is not None and 1 <= first_digit <= 8:
                allowed.extend(digit_ids)
            elif first_digit == 9:
                allowed.extend(digit_ids[:9])
            allowed.extend([separator_id] if field_index < 2 else [int(eos_token_id), separator_id])
        elif state == "number_done":
            allowed = [separator_id] if field_index < 2 else [int(eos_token_id), separator_id]
        elif state == "land":
            allowed = [land_tail[land_index]] if land_index < len(land_tail) else [int(eos_token_id)]
        else:  # pragma: no cover - internal state invariant
            raise RuntimeError(f"Unknown AerialVLA grammar state: {state}")
        allowed = [token_id for token_id in allowed if token_id not in forbidden]
        if not allowed:
            raise RuntimeError(f"AerialVLA grammar has no allowed tokens in state {state!r}")
        constrained = torch.full_like(logits, float("-inf"))
        constrained[:, allowed] = logits[:, allowed]
        next_ids = constrained.argmax(dim=-1)
        selected = int(next_ids[0].item())

        if state == "leading_separator":
            state = "first_digit"
        elif state == "first_digit":
            first_digit = digit_ids.index(selected)
            state = "number_tail"
        elif state in ("number_tail", "number_done"):
            if selected in digit_ids:
                value = int(first_digit) * 10 + digit_ids.index(selected)
                if not 0 <= value < int(num_bins):  # pragma: no cover - guarded by allowed
                    raise RuntimeError(f"Generated out-of-range AerialVLA bin: {value}")
                state = "number_done"
            elif selected == separator_id:
                if field_index < 2:
                    field_index += 1
                    first_digit = None
                    state = "first_digit"
                else:
                    state = "land"
                    land_index = 0
        elif state == "land":
            if land_index < len(land_tail):
                land_index += 1

        eos_fill = torch.full_like(next_ids, int(eos_token_id))
        next_ids = torch.where(finished, eos_fill, next_ids)
        generated = torch.cat([generated, next_ids[:, None]], dim=1)
        mask = torch.cat([mask, torch.ones_like(next_ids[:, None])], dim=1)
        finished |= next_ids.eq(int(eos_token_id))
        if bool(finished.all()):
            break
    return generated


class AerialVLABackend(ModelBackend):
    name = "aerialvla"

    def __init__(
        self,
        repo_root: str,
        ckpt_dir: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        adapter_dir: str | None = None,
    ):
        self.repo_root = Path(repo_root).resolve()
        self.ckpt_dir = Path(ckpt_dir).resolve()
        self.adapter_dir = (
            Path(adapter_dir).resolve() if adapter_dir else self.ckpt_dir / "lora"
        )
        self.device = device
        self.dtype_name = dtype
        for path in (self.repo_root, self.ckpt_dir / "openvla-7b", self.adapter_dir):
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
        codec = __import__(
            "src.aerialvla_action_codec", fromlist=["NUM_BINS", "parse_aerialvla_action_text"]
        )
        self.num_bins = int(codec.NUM_BINS)
        self.parse_action = codec.parse_aerialvla_action_text
        self.generate_no_cache = greedy_generate_no_cache
        base = self.ckpt_dir / "openvla-7b"
        lora = self.adapter_dir
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
        # Capture this interval before resize_token_embeddings changes the
        # padded vocabulary size in the in-memory config.  These ids encode
        # the base OpenVLA robot actions, not our three text bins.
        self.forbidden_action_token_ids = original_openvla_action_token_ids(model.config)
        model.resize_token_embeddings(len(self.tokenizer))
        self.model = PeftModel.from_pretrained(model, lora).to(device)
        self.model.eval()
        self.episode_id = None

    def health(self):
        return {
            "status": "ok",
            "backend": self.name,
            "checkpoint": str(self.ckpt_dir),
            "adapter": str(self.adapter_dir),
            "device": self.device,
            "dtype": self.dtype_name,
            "native_chunk_size": 1,
            "action_frame": "body",
            "action_semantics": "[forward,0,down,d_yaw,LAND]",
            "action_codec": "src.aerialvla_action_codec.parse_aerialvla_action_text",
            "generation": "greedy_no_cache_constrained",
            "forbidden_action_token_range": [
                self.forbidden_action_token_ids[0],
                self.forbidden_action_token_ids[-1],
            ],
            "strict_action_format": True,
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
            output_ids = self.generate_no_cache(
                self.model,
                input_ids=text["input_ids"],
                attention_mask=text["attention_mask"],
                pixel_values=pixels,
                max_new_tokens=20,
                eos_token_id=self.tokenizer.eos_token_id,
                forbidden_token_ids=self.forbidden_action_token_ids,
                tokenizer=self.tokenizer,
                num_bins=self.num_bins,
            )
        decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=False)
        validate_aerialvla_action_text(decoded, self.num_bins)
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
