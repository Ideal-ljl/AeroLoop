import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from uav_eval.cameras import CameraSpec
from uav_eval.policies.http import HttpPolicy
from uav_eval.types import EpisodeSpec, Observation, PolicyInput, Pose


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        size = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(size))
        Handler.last_payload = payload
        body = json.dumps({"actions": [[1, 2, 3, 0.25]], "metadata": {"ok": True}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


class HttpPolicyTest(unittest.TestCase):
    def test_canonical_contract_and_four_column_compatibility(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            episode = EpisodeSpec("ep", "mock", "go", Pose(0, 0, 0, 0), (1, 0, 0), 1)
            obs = Observation(None, Pose(0, 0, 0, 0), (0, 0, 0, 0), 0)
            policy = HttpPolicy(f"http://127.0.0.1:{server.server_port}/v1/predict")
            policy.reset(episode)
            chunk = policy.predict(PolicyInput(episode, obs))
            self.assertEqual(chunk.actions[0].as_list(), [1.0, 2.0, 3.0, 0.0, 0.25])
            self.assertEqual(Handler.last_payload["image_base64"], None)
            self.assertEqual(Handler.last_payload["instruction"], "go")
        finally:
            server.shutdown()
            server.server_close()

    def test_named_views_and_auxiliary_state_are_sent(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            episode = EpisodeSpec("ep", "mock", "go", Pose(0, 0, 0, 0), (1, 0, 0), 1)
            obs = Observation(
                "front-frame",
                Pose(0, 0, 0, 0),
                (0, 0, 0, 0),
                0,
                images={"front": "front-frame", "down": "down-frame"},
                auxiliary_state={"velocity": [1, 0, 0]},
                camera_specs={"front": CameraSpec("front", width=640, height=480, fov=90)},
            )
            policy = HttpPolicy(f"http://127.0.0.1:{server.server_port}/v1/predict", views=["front", "down"])
            with patch("uav_eval.policies.http._encode_rgb_png", side_effect=lambda image: f"png:{image}"):
                policy.predict(PolicyInput(episode, obs))
            self.assertEqual(
                Handler.last_payload["images_base64"],
                {"front": "png:front-frame", "down": "png:down-frame"},
            )
            self.assertEqual(Handler.last_payload["image_base64"], "png:front-frame")
            self.assertEqual(Handler.last_payload["auxiliary_state"]["velocity"], [1, 0, 0])
            self.assertEqual(Handler.last_payload["camera_specs"]["front"]["width"], 640)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
