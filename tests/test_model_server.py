import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import error, request

from uav_eval.policies.http import HttpPolicy
from uav_eval.server import ModelBackend, PredictRequest, create_server
from uav_eval.server.backends.openuav import missing_navigation_weights, normalize_checkpoint_state
from uav_eval.types import EpisodeSpec, Observation, PolicyInput, Pose


class FakeBackend(ModelBackend):
    name = "fake"

    def __init__(self):
        self.resets = []
        self.predicts = []

    def reset(self, episode_id, instruction, env_name):
        self.resets.append((episode_id, instruction, env_name))
        return {"backend_reset": True}

    def predict(self, request: PredictRequest):
        self.predicts.append(request)
        return {"actions": [[1, 0, 0, 0, 0]], "metadata": {"backend": "fake"}}


class ModelServerTest(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend()
        self.server = create_server(self.backend, "127.0.0.1", 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.root = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def test_health_reset_and_predict_with_http_policy(self):
        with request.urlopen(f"{self.root}/health") as response:
            health = json.loads(response.read())
        self.assertEqual(health["backend"], "fake")

        episode = EpisodeSpec("ep", "mock", "go", Pose(0, 0, 0, 0), (1, 0, 0), 1)
        policy = HttpPolicy(f"{self.root}/v1/predict", reset_url=f"{self.root}/v1/reset")
        policy.reset(episode)
        obs = Observation(None, Pose(0, 0, 0, 0), (0, 0, 0, 0), 0)
        chunk = policy.predict(PolicyInput(episode, obs))
        self.assertEqual(chunk.actions[0].as_list(), [1.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(self.backend.resets, [("ep", "go", "mock")])
        self.assertEqual(self.backend.predicts[0].state, (0.0, 0.0, 0.0, 0.0))

    def test_bad_request_returns_400(self):
        payload = json.dumps({"episode_id": "ep", "state": [0], "pose": [0]}).encode()
        req = request.Request(
            f"{self.root}/v1/predict", data=payload, headers={"Content-Type": "application/json"}, method="POST"
        )
        with self.assertRaises(error.HTTPError) as caught:
            request.urlopen(req)
        self.assertEqual(caught.exception.code, 400)

    def test_openuav_checkpoint_prefix_normalization(self):
        state = {
            "base_model.model.model.mm_projector.weight": object(),
            "base_model.model.waypoints_fc.0.weight": object(),
        }
        normalized = normalize_checkpoint_state(
            state, {"model.mm_projector.weight", "waypoints_fc.0.weight"}
        )
        self.assertEqual(
            set(normalized), {"model.mm_projector.weight", "waypoints_fc.0.weight"}
        )

    def test_openuav_incomplete_checkpoint_is_detected(self):
        keys = ["model.mm_projector.weight", "model.vlm_att_query"]
        self.assertEqual(
            missing_navigation_weights(keys),
            ["embed_tokens", "waypoint_emb", "waypoints_fc", "waypoints_output", "history_preprocessor"],
        )


if __name__ == "__main__":
    unittest.main()
