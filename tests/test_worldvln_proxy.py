import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from aeroloop.server.base import PredictRequest
from aeroloop.server.backends.worldvln import WorldVLNProxyBackend


class UpstreamHandler(BaseHTTPRequestHandler):
    calls = []

    def do_GET(self):  # noqa: N802
        self._send({"status": "ok", "step": 2})

    def do_POST(self):  # noqa: N802
        size = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(size))
        self.__class__.calls.append(payload)
        self._send(
            {
                "actions": [
                    [100, 0, 0, 0, 90, 0],
                    [0, 200, 0, 0, 0, 0],
                ],
                "segment_index": len(self.__class__.calls) - 1,
                "done": False,
            }
        )

    def _send(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


class DoneUpstreamHandler(UpstreamHandler):
    def do_POST(self):  # noqa: N802
        size = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(size))
        self.__class__.calls.append(payload)
        self._send({"actions": [], "segment_index": -1, "done": True})


def predict_request(step):
    return PredictRequest("ep", "mock", "go", step, (0, 0, 0, 0), (0, 0, 0, 0), f"frame-{step}")


class WorldVLNProxyTest(unittest.TestCase):
    def test_default_stop_threshold_is_half_a_metre(self):
        backend = WorldVLNProxyBackend("http://127.0.0.1:1")
        self.assertEqual(backend.stop_speed_threshold, 0.5)

    def test_stop_threshold_is_applied_after_centimetres_to_metres_conversion(self):
        backend = WorldVLNProxyBackend("http://127.0.0.1:1", stop_speed_threshold=0.5)
        actions = backend._canonicalize_chunk(
            [[40, 0, 0, 0, 0, 0], [30, 0, 0, 0, 0, 0], [20, 0, 0, 0, 0, 0]]
        )
        self.assertEqual(actions[-1][-1], 1.0)
        self.assertEqual(actions[0][0], 0.4)

    def test_motion_above_half_a_metre_does_not_stop(self):
        backend = WorldVLNProxyBackend("http://127.0.0.1:1", stop_speed_threshold=0.5)
        actions = backend._canonicalize_chunk(
            [[60, 0, 0, 0, 0, 0], [70, 0, 0, 0, 0, 0], [80, 0, 0, 0, 0, 0]]
        )
        self.assertEqual(actions[-1][-1], 0.0)

    def test_reset_reuses_native_session_to_avoid_stream_accumulation(self):
        backend = WorldVLNProxyBackend("http://127.0.0.1:1")
        first = backend.reset("ep-1", "go", "mock")
        second = backend.reset("ep-2", "turn", "mock")
        self.assertEqual(first["upstream_session_id"], second["upstream_session_id"])
        self.assertEqual(backend.episode_id, "ep-2")
        self.assertEqual(backend.instruction, "turn")
        self.assertEqual(backend.segment_index, -1)

    def test_segment_actions_are_cached_and_units_are_converted(self):
        UpstreamHandler.calls = []
        server = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            backend = WorldVLNProxyBackend(f"http://127.0.0.1:{server.server_port}", stop_speed_threshold=0)
            backend.reset("ep", "go", "mock")
            first = backend.predict(predict_request(0))
            second = backend.predict(predict_request(1))
            third = backend.predict(predict_request(2))
            self.assertEqual(len(UpstreamHandler.calls), 2)
            self.assertEqual(len(UpstreamHandler.calls[0]["images_base64"]), 1)
            self.assertTrue(UpstreamHandler.calls[0]["reset_session"])
            self.assertEqual(len(UpstreamHandler.calls[1]["images_base64"]), 2)
            self.assertEqual(first["actions"][0][:4], [1.0, 0.0, 0.0, 1.5707963267948966])
            self.assertEqual(second["actions"][0][:4], [0.0, 2.0, 0.0, 0.0])
            self.assertTrue(first["metadata"]["native_replan"])
            self.assertFalse(second["metadata"]["native_replan"])
            self.assertTrue(third["metadata"]["native_replan"])
        finally:
            server.shutdown()
            server.server_close()

    def test_native_done_without_actions_becomes_explicit_stop(self):
        DoneUpstreamHandler.calls = []
        server = ThreadingHTTPServer(("127.0.0.1", 0), DoneUpstreamHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            backend = WorldVLNProxyBackend(f"http://127.0.0.1:{server.server_port}")
            backend.reset("ep", "go", "mock")
            result = backend.predict(predict_request(0))
            self.assertEqual(result["actions"], [[0.0, 0.0, 0.0, 0.0, 1.0]])
            self.assertTrue(result["metadata"]["native_replan"])
            self.assertTrue(result["metadata"]["upstream"]["done"])
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
