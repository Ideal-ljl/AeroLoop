from __future__ import annotations

import json
import math
import uuid
from collections import deque
from urllib import request as urlrequest

from ..base import ModelBackend, PredictRequest


class WorldVLNProxyBackend(ModelBackend):
    """Canonical proxy for WorldVLN's native segmented inference server."""

    name = "worldvln"

    def __init__(
        self,
        upstream_url: str = "http://127.0.0.1:8001",
        timeout_s: float = 600.0,
        action_head_mode: str = "tsformer_latent",
        stop_speed_threshold: float = 0.005,
        allow_future_last_segment: bool = True,
    ):
        self.upstream_url = upstream_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.action_head_mode = action_head_mode
        self.stop_speed_threshold = float(stop_speed_threshold)
        self.allow_future_last_segment = bool(allow_future_last_segment)
        self.session_id = None
        self.episode_id = None
        self.instruction = ""
        self.pending_actions = deque()
        self.pending_frames = []
        self.segment_index = -1

    def _request(self, method: str, path: str, payload=None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(
            f"{self.upstream_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"} if data is not None else {},
            method=method,
        )
        with urlrequest.urlopen(req, timeout=self.timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    def health(self):
        upstream = self._request("GET", "/health")
        return {
            "status": "ok",
            "backend": self.name,
            "upstream_url": self.upstream_url,
            "upstream": upstream,
            "native_segmented_execution": True,
            "action_head_mode": self.action_head_mode,
        }

    def reset(self, episode_id: str, instruction: str, env_name: str):
        self.episode_id = episode_id
        self.instruction = instruction
        self.session_id = uuid.uuid4().hex
        self.pending_actions.clear()
        self.pending_frames.clear()
        self.segment_index = -1
        return {"episode_id": episode_id, "upstream_session_id": self.session_id}

    def _canonicalize_chunk(self, rows):
        translations = []
        actions = []
        for row in rows:
            if len(row) < 6:
                raise ValueError(f"WorldVLN upstream action needs 6 values, got {row}")
            dx, dy, dz = (float(row[0]) / 100.0, float(row[1]) / 100.0, float(row[2]) / 100.0)
            translations.append(math.sqrt(dx * dx + dy * dy + dz * dz))
            d_yaw = float(row[4]) * math.pi / 180.0
            actions.append([dx, dy, dz, d_yaw, 0.0])
        if actions and sum(translations[-3:]) / min(3, len(translations)) < self.stop_speed_threshold:
            actions[-1][-1] = 1.0
        return actions

    def _replan(self):
        payload = {
            "session_id": self.session_id,
            "images_base64": list(self.pending_frames),
            "allow_future_segments": True,
            "allow_future_last_segment": self.allow_future_last_segment,
            "action_head_mode": self.action_head_mode,
        }
        if self.segment_index < 0:
            payload["instruction"] = self.instruction
            payload["reset_session"] = True
        response = self._request("POST", "/v1/predict_delta_actions", payload)
        rows = response.get("actions") or []
        if not rows:
            raise RuntimeError(f"WorldVLN upstream emitted no action segment: {response}")
        self.pending_actions.extend(self._canonicalize_chunk(rows))
        self.pending_frames.clear()
        self.segment_index = int(response.get("segment_index", self.segment_index + 1))
        return response

    def predict(self, request: PredictRequest):
        if self.episode_id != request.episode_id:
            self.reset(request.episode_id, request.instruction, request.env_name)
        if not request.image_base64:
            raise ValueError("WorldVLN requires image_base64")
        self.pending_frames.append(request.image_base64)
        replanned = False
        upstream_metadata = None
        if not self.pending_actions:
            upstream_metadata = self._replan()
            replanned = True
        action = self.pending_actions.popleft()
        return {
            "actions": [action],
            "metadata": {
                "model": self.name,
                "native_replan": replanned,
                "native_segment_index": self.segment_index,
                "cached_actions_remaining": len(self.pending_actions),
                "buffered_real_frames": len(self.pending_frames),
                "upstream": upstream_metadata,
            },
        }
