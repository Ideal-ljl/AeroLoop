"""Dependency-free reference implementation of the canonical policy API."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _json(self):
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length) or b"{}")

    def _send(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        payload = self._json()
        if self.path == "/v1/reset":
            self._send({"ok": True, "episode_id": payload.get("episode_id")})
            return
        if self.path == "/v1/predict":
            state = payload.get("state", [0, 0, 0, 0])
            stop = 1.0 if float(state[0]) >= 10.0 else 0.0
            action = [0.0, 0.0, 0.0, 0.0, 1.0] if stop else [1.0, 0.0, 0.0, 0.0, 0.0]
            self._send({"actions": [action], "metadata": {"model": "mock-http"}})
            return
        self._send({"error": "not found"}, status=404)

    def log_message(self, *_args):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()
    print(f"mock policy listening on http://{args.host}:{args.port}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
