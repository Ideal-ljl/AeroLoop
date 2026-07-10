from __future__ import annotations

import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping

from .base import ModelBackend, PredictRequest


def _validate_response(response: Mapping[str, Any]) -> dict[str, Any]:
    actions = response.get("actions")
    if not isinstance(actions, list) or not actions:
        raise ValueError("backend response must contain a non-empty actions list")
    for index, row in enumerate(actions):
        if not isinstance(row, (list, tuple)) or len(row) not in (4, 5):
            raise ValueError(f"action {index} must contain 4 or 5 values")
        try:
            numeric = [float(value) for value in row]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"action {index} contains a non-numeric value") from exc
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError(f"action {index} contains NaN or infinity")
    return {
        "actions": [[float(value) for value in row] for row in actions],
        "metadata": dict(response.get("metadata") or {}),
    }


def create_server(backend: ModelBackend, host: str = "127.0.0.1", port: int = 18080) -> ThreadingHTTPServer:
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _read_json(self) -> dict:
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0:
                return {}
            value = json.loads(self.rfile.read(size).decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("request body must be a JSON object")
            return value

        def _send(self, status: int, payload: Mapping[str, Any]) -> None:
            body = json.dumps(dict(payload), ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path != "/health":
                self._send(404, {"error": "not found"})
                return
            try:
                with lock:
                    health = dict(backend.health())
                health.setdefault("status", "ok")
                self._send(200, health)
            except Exception as exc:
                self._send(500, {"status": "error", "error": f"{type(exc).__name__}: {exc}"})

        def do_POST(self):  # noqa: N802
            try:
                payload = self._read_json()
                with lock:
                    if self.path == "/v1/reset":
                        episode_id = str(payload.get("episode_id", "")).strip()
                        if not episode_id:
                            raise ValueError("episode_id is required")
                        result = backend.reset(
                            episode_id=episode_id,
                            instruction=str(payload.get("instruction", "")),
                            env_name=str(payload.get("env_name", "")),
                        )
                        self._send(200, {"ok": True, **dict(result or {})})
                        return
                    if self.path == "/v1/predict":
                        response = _validate_response(backend.predict(PredictRequest.from_mapping(payload)))
                        self._send(200, response)
                        return
                self._send(404, {"error": "not found"})
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                self._send(400, {"error": f"{type(exc).__name__}: {exc}"})
            except Exception as exc:
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, format, *args):
            print(f"[model-server] {self.address_string()} {format % args}")

    server = ThreadingHTTPServer((host, int(port)), Handler)
    server.backend = backend  # type: ignore[attr-defined]
    return server


def serve(backend: ModelBackend, host: str = "0.0.0.0", port: int = 18080) -> None:
    server = create_server(backend, host, port)
    print(f"{backend.name} listening on http://{host}:{server.server_port}")
    try:
        server.serve_forever()
    finally:
        server.server_close()
        backend.close()
