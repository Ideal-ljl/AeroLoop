from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rewrite_path(value: str | Path, root: str | Path | None, rewrites: Mapping[str, str] | None) -> Path:
    text = str(value)
    for old, new in (rewrites or {}).items():
        if text.startswith(str(old)):
            text = str(new) + text[len(str(old)) :]
            break
    path = Path(text)
    if root is not None and not path.is_absolute():
        path = Path(root) / path
    return path


def path_length(points: Iterable[Sequence[float]]) -> float:
    rows = [tuple(map(float, point[:3])) for point in points]
    return sum(math.dist(before, after) for before, after in zip(rows, rows[1:]))


def yaw_from_quaternion(values: Sequence[float]) -> float:
    if len(values) < 4:
        return 0.0
    x, y, z, w = map(float, values[:4])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
