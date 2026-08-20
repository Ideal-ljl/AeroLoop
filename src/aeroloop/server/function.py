from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ..extensions import import_object
from .base import ModelBackend, PredictRequest


def _select(request: PredictRequest, selector: str) -> Any:
    """Resolve a stable protocol selector into a model-function value."""

    if selector == "image":
        return request.decode_rgb()
    if selector.startswith("images."):
        return request.decode_rgb(selector.split(".", 1)[1])
    value: Any = request
    for part in selector.split("."):
        if isinstance(value, Mapping):
            value = value[part]
        else:
            value = getattr(value, part)
    return value


@dataclass
class FunctionBackend(ModelBackend):
    """Expose an existing inference function without writing a backend class.

    ``inputs`` maps the inference function's argument names to canonical request
    selectors, for example ``{"prompt": "instruction", "rgb": "image"}``.
    """

    entrypoint: str
    inputs: Mapping[str, str]
    reset_entrypoint: str | None = None
    static_kwargs: Mapping[str, Any] = field(default_factory=dict)
    actions_key: str = "actions"
    name: str = "function"

    def __post_init__(self) -> None:
        self.function: Callable[..., Any] = import_object(self.entrypoint)
        self.reset_function: Callable[..., Any] | None = (
            import_object(self.reset_entrypoint) if self.reset_entrypoint else None
        )

    def reset(self, episode_id: str, instruction: str, env_name: str):
        if self.reset_function is None:
            return None
        result = self.reset_function(episode_id=episode_id, instruction=instruction, env_name=env_name)
        return result if isinstance(result, Mapping) else None

    def predict(self, request: PredictRequest):
        kwargs = {name: _select(request, selector) for name, selector in self.inputs.items()}
        kwargs.update(self.static_kwargs)
        result = self.function(**kwargs)
        if isinstance(result, Mapping):
            actions = result.get(self.actions_key)
            metadata = dict(result.get("metadata") or {})
        else:
            actions = result
            metadata = {}
        return {"actions": actions, "metadata": metadata}
