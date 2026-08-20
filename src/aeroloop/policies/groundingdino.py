from __future__ import annotations

from collections import deque
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..protocols import PolicyAdapter
from ..types import ActionChunk, CanonicalAction, EpisodeSpec, PolicyInput, Transition


def _local_model_directory(value: str | Path) -> Path:
    path = Path(value).resolve()
    if (path / "config.json").is_file():
        return path
    snapshots = sorted(candidate.parent for candidate in path.glob("snapshots/*/config.json"))
    if len(snapshots) == 1:
        return snapshots[0]
    if not path.exists():
        raise FileNotFoundError(f"GroundingDINO model path not found: {path}")
    raise FileNotFoundError(
        f"GroundingDINO config.json not found under {path}; provide a materialized model directory "
        "or a Hugging Face cache containing exactly one snapshot"
    )


def qualifying_detections(
    boxes: Sequence[Sequence[float]],
    scores: Sequence[float],
    labels: Sequence[str],
    depth,
    *,
    image_width: int,
    image_height: int,
    max_depth: float = 18.0,
    max_box_width_ratio: float = 0.6,
    max_box_height_ratio: float = 0.5,
) -> list[dict[str, Any]]:
    """Apply the original OpenUAV GroundingDINO box and center-depth rule."""

    accepted = []
    for box, score, label in zip(boxes, scores, labels):
        x0, y0, x1, y1 = map(float, box[:4])
        if not all(math.isfinite(value) for value in (x0, y0, x1, y1, float(score))):
            continue
        if (x1 - x0) / image_width > max_box_width_ratio:
            continue
        if (y1 - y0) / image_height > max_box_height_ratio:
            continue
        center_x = min(max(int((x0 + x1) / 2), 0), image_width - 1)
        center_y = min(max(int((y0 + y1) / 2), 0), image_height - 1)
        distance = float(depth[center_y, center_x])
        if not math.isfinite(distance) or distance >= max_depth:
            continue
        accepted.append(
            {
                "box": [x0, y0, x1, y1],
                "score": float(score),
                "label": str(label),
                "center_depth": distance,
            }
        )
    return accepted


class TransformersGroundingDinoDetector:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        box_threshold: float = 0.6,
        text_threshold: float = 0.4,
        max_depth: float = 18.0,
        max_box_width_ratio: float = 0.6,
        max_box_height_ratio: float = 0.5,
    ):
        model_dir = _local_model_directory(model_path)
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "Transformers GroundingDINO requires a model environment with a recent transformers release"
            ) from exc
        self.torch = torch
        self.device = device
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.max_depth = float(max_depth)
        self.max_box_width_ratio = float(max_box_width_ratio)
        self.max_box_height_ratio = float(max_box_height_ratio)
        self.processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_dir, local_files_only=True
        ).to(device)
        self.model.eval()

    def detect(self, image, prompt: str, depth) -> list[dict[str, Any]]:
        height, width = image.shape[:2]
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.get("input_ids"),
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(height, width)],
        )[0]
        boxes = result["boxes"].detach().float().cpu().tolist()
        scores = result["scores"].detach().float().cpu().tolist()
        labels = result.get("text_labels") or result.get("labels") or [""] * len(boxes)
        return qualifying_detections(
            boxes,
            scores,
            labels,
            depth,
            image_width=width,
            image_height=height,
            max_depth=self.max_depth,
            max_box_width_ratio=self.max_box_width_ratio,
            max_box_height_ratio=self.max_box_height_ratio,
        )


class OriginalGroundingDinoDetector:
    def __init__(
        self,
        source_root: str,
        config_path: str,
        model_path: str,
        device: str = "cuda:0",
        box_threshold: float = 0.6,
        text_threshold: float = 0.4,
        max_depth: float = 18.0,
        max_box_width_ratio: float = 0.6,
        max_box_height_ratio: float = 0.5,
    ):
        root = Path(source_root).resolve()
        config = Path(config_path).resolve()
        checkpoint = Path(model_path).resolve()
        for path in (root, config, checkpoint):
            if not path.exists():
                raise FileNotFoundError(path)
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        try:
            import torch
            import groundingdino.datasets.transforms as transforms
            from groundingdino.util import box_ops
            from groundingdino.util.inference import load_model, predict
        except ImportError as exc:
            raise RuntimeError(f"cannot import original GroundingDINO from {root}") from exc
        self.torch = torch
        self.transforms = transforms
        self.box_ops = box_ops
        self.predict_fn = predict
        self.device = device
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.max_depth = float(max_depth)
        self.max_box_width_ratio = float(max_box_width_ratio)
        self.max_box_height_ratio = float(max_box_height_ratio)
        self.model = load_model(str(config), str(checkpoint), device=device).to(device)

    def detect(self, image, prompt: str, depth) -> list[dict[str, Any]]:
        from PIL import Image

        height, width = image.shape[:2]
        transform = self.transforms.Compose(
            [
                self.transforms.RandomResize([800], max_size=1333),
                self.transforms.ToTensor(),
                self.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        transformed, _ = transform(Image.fromarray(image), None)
        boxes, scores, labels = self.predict_fn(
            model=self.model,
            image=transformed,
            caption=prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device,
        )
        boxes = self.box_ops.box_cxcywh_to_xyxy(boxes) * self.torch.tensor([width, height, width, height])
        return qualifying_detections(
            boxes.detach().float().cpu().tolist(),
            scores.detach().float().cpu().tolist(),
            labels,
            depth,
            image_width=width,
            image_height=height,
            max_depth=self.max_depth,
            max_box_width_ratio=self.max_box_width_ratio,
            max_box_height_ratio=self.max_box_height_ratio,
        )


def build_groundingdino_detector(config: Mapping[str, Any]):
    values = dict(config)
    implementation = str(values.pop("implementation", "transformers"))
    if implementation == "transformers":
        return TransformersGroundingDinoDetector(**values)
    if implementation == "original":
        return OriginalGroundingDinoDetector(**values)
    raise ValueError("GroundingDINO implementation must be 'transformers' or 'original'")


class GroundingDinoStopPolicy(PolicyAdapter):
    name = "groundingdino_stop"

    def __init__(
        self,
        base_policy: PolicyAdapter,
        detector: Any | None = None,
        detector_config: Mapping[str, Any] | None = None,
        views: Sequence[str] = ("front",),
        prompt_metadata_key: str = "grounding_prompt",
        fallback_to_instruction: bool = False,
        min_step: int = 1,
        reuse_base_chunk: bool = True,
    ):
        if detector is None and detector_config is None:
            raise ValueError("GroundingDINO stop policy requires detector or detector_config")
        self.base_policy = base_policy
        self.detector = detector or build_groundingdino_detector(detector_config or {})
        self.views = tuple(str(view) for view in views)
        self.prompt_metadata_key = str(prompt_metadata_key)
        self.fallback_to_instruction = bool(fallback_to_instruction)
        self.min_step = int(min_step)
        self.reuse_base_chunk = bool(reuse_base_chunk)
        self.episode: EpisodeSpec | None = None
        self._pending_actions: deque[tuple[CanonicalAction, int, int, dict[str, Any]]] = deque()

    def reset(self, episode: EpisodeSpec) -> None:
        self.episode = episode
        self._pending_actions.clear()
        self.base_policy.reset(episode)

    def _prompt(self, episode: EpisodeSpec) -> str:
        value = str(episode.metadata.get(self.prompt_metadata_key, "")).strip()
        if not value and self.fallback_to_instruction:
            value = episode.instruction.strip()
        if not value:
            raise ValueError(f"episode {episode.episode_id!r} has no GroundingDINO target prompt")
        return value

    def predict(self, policy_input: PolicyInput) -> ActionChunk:
        observation = policy_input.observation
        if observation.step_index >= self.min_step:
            depth_maps = observation.info.get("depth") or {}
            prompt = self._prompt(policy_input.episode)
            detections = []
            for view in self.views:
                if view not in observation.images:
                    raise ValueError(f"GroundingDINO view {view!r} is unavailable")
                if view not in depth_maps:
                    raise ValueError(f"GroundingDINO depth for view {view!r} is unavailable")
                rows = self.detector.detect(observation.images[view], prompt, depth_maps[view])
                detections.extend({"view": view, **dict(row)} for row in rows)
            if detections:
                self._pending_actions.clear()
                return ActionChunk(
                    (CanonicalAction(0.0, 0.0, 0.0, 0.0, 1.0),),
                    {
                        "model": self.name,
                        "grounding_prompt": prompt,
                        "grounding_detections": detections,
                        "grounding_stop": True,
                    },
                )
        if not self.reuse_base_chunk:
            return self.base_policy.predict(policy_input)

        if not self._pending_actions:
            chunk = self.base_policy.predict(policy_input)
            chunk_size = len(chunk.actions)
            metadata = dict(chunk.metadata)
            for action_index, action in enumerate(chunk.actions):
                self._pending_actions.append((action, action_index, chunk_size, metadata))

        action, action_index, chunk_size, base_metadata = self._pending_actions.popleft()
        if action.stop > 0.0:
            self._pending_actions.clear()
        metadata = {
            **base_metadata,
            "grounding_stop": False,
            "base_action_index": action_index,
            "base_chunk_size": chunk_size,
            "cached_base_action": action_index > 0,
            "cached_actions_remaining": len(self._pending_actions),
        }
        return ActionChunk((action,), metadata)

    def on_action_executed(self, action: CanonicalAction, transition: Transition) -> None:
        self.base_policy.on_action_executed(action, transition)

    def close(self) -> None:
        self._pending_actions.clear()
        self.base_policy.close()
