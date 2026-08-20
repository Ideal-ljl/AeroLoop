from __future__ import annotations

import html
import base64
import json
import mimetypes
from pathlib import Path
from itertools import islice
from typing import Iterable, Sequence

from ..types import EpisodeSpec


def _sample_images(episode: EpisodeSpec, count: int = 3) -> list[Path]:
    root = Path(str(episode.metadata.get("trajectory_dir", "")))
    if not root.is_dir():
        return []
    candidates = (
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    return list(islice(candidates, count))


def _image_gallery(episode: EpisodeSpec) -> str:
    images = []
    for path in _sample_images(episode):
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        images.append(
            f'<figure><img src="data:{mime};base64,{encoded}">'
            f"<figcaption>{html.escape(path.name)}</figcaption></figure>"
        )
    return f'<div class="gallery">{"".join(images)}</div>' if images else ""


def _polyline(points: Sequence[Sequence[float]], width: int = 480, height: int = 280) -> str:
    if not points:
        return ""
    xs, ys = [float(p[0]) for p in points], [float(p[1]) for p in points]
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
    scale = min((width - 32) / max(max_x - min_x, 1e-8), (height - 32) / max(max_y - min_y, 1e-8))
    coords = [f"{16 + (x-min_x)*scale:.1f},{height-16-(y-min_y)*scale:.1f}" for x, y in zip(xs, ys)]
    return " ".join(coords)


def write_dataset_html(episodes: Iterable[EpisodeSpec], output: str | Path, limit: int = 100) -> Path:
    cards = []
    rows = list(episodes)[: int(limit)]
    for episode in rows:
        trajectory = episode.metadata.get("trajectory") or [episode.start_pose.xyz(), episode.target_position]
        points = _polyline(trajectory)
        metadata = {k: v for k, v in episode.metadata.items() if k not in {"raw", "trajectory"}}
        cards.append(
            f"<article><h2>{html.escape(episode.episode_id)}</h2>"
            f"<p><b>{html.escape(episode.env_name)}</b> · {len(trajectory)} points · "
            f"{episode.reference_path_length:.1f} m</p>"
            f"<p>{html.escape(episode.instruction)}</p>"
            f'<svg viewBox="0 0 480 280"><polyline points="{points}"/></svg>'
            f"{_image_gallery(episode)}"
            "<details><summary>metadata</summary><pre>"
            f"{html.escape(json.dumps(metadata, ensure_ascii=False, indent=2))}"
            "</pre></details>"
            "</article>"
        )
    document = f"""<!doctype html><html><head><meta charset="utf-8"><title>AeroLoop dataset</title>
<style>
body{{font:14px system-ui;margin:24px;background:#f4f6f8}}
main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(500px,1fr));gap:18px}}
article{{background:white;padding:18px;border-radius:12px;box-shadow:0 2px 12px #0001}}
svg{{width:100%;background:#101820;border-radius:8px}}
polyline{{fill:none;stroke:#39d98a;stroke-width:3}}pre{{overflow:auto}}
.gallery{{display:flex;gap:8px;overflow:auto;margin-top:10px}}figure{{margin:0}}img{{height:120px;border-radius:6px}}
figcaption{{font-size:11px;color:#667}}
</style></head>
<body><h1>AeroLoop dataset preview ({len(rows)} episodes)</h1>
<main>{''.join(cards)}</main></body></html>"""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    return path
