import json
import importlib.util
import threading
import unittest
from types import SimpleNamespace
from urllib import error, request

from aeroloop.policies.http import HttpPolicy
from aeroloop.server import ModelBackend, PredictRequest, create_server, heading_delta_from_translation
from aeroloop.server.backends.aerialvla import (
    build_aerialvla_prompt,
    greedy_generate_no_cache,
    original_openvla_action_token_ids,
    validate_aerialvla_action_text,
)
from aeroloop.server.backends.openuav import (
    cumulative_waypoints_to_actions,
    missing_navigation_weights,
    normalize_checkpoint_state,
    normalize_assistant_stage,
    waypoint_label_present,
)
from aeroloop.server.backends.omninav import (
    build_omninav_prompt,
    cumulative_waypoints_to_actions as omninav_waypoints_to_actions,
    history_positions_in_current_body,
    select_history_indices,
)
from aeroloop.server.backends.pi0 import pi0_actions_to_canonical
from aeroloop.server.function import FunctionBackend
from aeroloop.types import EpisodeSpec, Observation, PolicyInput, Pose


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
        chunk = policy.predict(PolicyInput(episode, obs, policy_context={"assistant_stage": "right"}))
        self.assertEqual(chunk.actions[0].as_list(), [1.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(self.backend.resets, [("ep", "go", "mock")])
        self.assertEqual(self.backend.predicts[0].state, (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(self.backend.predicts[0].policy_context["assistant_stage"], "right")

    def test_openuav_assistant_stage_is_strictly_normalized(self):
        self.assertEqual(normalize_assistant_stage("takeoff"), "take off")
        self.assertEqual(normalize_assistant_stage("LEFT"), "left")
        with self.assertRaisesRegex(ValueError, "unsupported OpenUAV assistant stage"):
            normalize_assistant_stage("fly through the wall")

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
        normalized = normalize_checkpoint_state(state, {"model.mm_projector.weight", "waypoints_fc.0.weight"})
        self.assertEqual(set(normalized), {"model.mm_projector.weight", "waypoints_fc.0.weight"})

    def test_openuav_incomplete_checkpoint_is_detected(self):
        keys = ["model.mm_projector.weight", "model.vlm_att_query"]
        self.assertEqual(
            missing_navigation_weights(keys),
            ["embed_tokens", "waypoint_emb", "waypoints_fc", "waypoints_output", "history_preprocessor"],
        )

    def test_aerialvla_prompt_matches_training_format(self):
        self.assertEqual(build_aerialvla_prompt("  fly left  "), "<image>\nInstruction:fly left;\nAction: ")
        self.assertEqual(
            build_aerialvla_prompt("Instruction:fly left;"),
            "<image>\nInstruction:fly left;\nAction: ",
        )

    def test_aerialvla_blocks_original_openvla_action_tokens(self):
        config = SimpleNamespace(
            text_config=SimpleNamespace(vocab_size=32064),
            pad_to_multiple_of=64,
            n_action_bins=256,
        )
        token_ids = original_openvla_action_token_ids(config)
        self.assertEqual((token_ids[0], token_ids[-1], len(token_ids)), (31744, 31999, 256))

    def test_aerialvla_action_text_validation_is_strict(self):
        self.assertEqual(
            validate_aerialvla_action_text("<s> prompt\nAction:  50 39 49</s>"),
            (50, 39, 49, False),
        )
        self.assertEqual(
            validate_aerialvla_action_text("Action: 0 49 49 <LAND></s>"),
            (0, 49, 49, True),
        )
        for malformed in ("Action: 种타操", "Action: 50 39彦", "Action: 50 39", "Action: 99 49 49"):
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                validate_aerialvla_action_text(malformed)

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is required")
    def test_aerialvla_constrained_generation_always_completes_three_bins(self):
        import torch

        class FakeTokenizer:
            def encode(self, value, add_special_tokens=False):
                if value == " <LAND>":
                    return [10, 30, 31]
                digit = int(value)
                return [10, 20 + digit]

        class FakeModel:
            def __call__(self, input_ids, **kwargs):
                # Equal logits make unconstrained argmax choose token zero. The
                # grammar must still produce the canonical shortest action.
                logits = torch.zeros((1, input_ids.shape[1], 64))
                return SimpleNamespace(logits=logits)

        output = greedy_generate_no_cache(
            FakeModel(),
            input_ids=torch.tensor([[1, 5]]),
            attention_mask=torch.ones((1, 2), dtype=torch.long),
            pixel_values=torch.zeros((1, 3, 2, 2)),
            max_new_tokens=16,
            eos_token_id=2,
            forbidden_token_ids=(40,),
            tokenizer=FakeTokenizer(),
            num_bins=99,
        )
        self.assertEqual(output[0, 2:].tolist(), [10, 20, 10, 20, 10, 20, 2])

    def test_openuav_requires_waypoint_label_from_training_preprocess(self):
        self.assertTrue(waypoint_label_present([[1, -200, 2]], -200))
        self.assertFalse(waypoint_label_present([[1, 2, 3]], -200))

    def test_openuav_cumulative_trajectory_becomes_delta_chunk(self):
        actions = cumulative_waypoints_to_actions([[1, 0, 0], [1, 2, 0], [0, 2, -1]])
        self.assertEqual(actions[0], [1.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(actions[1][:3], [0.0, 2.0, 0.0])
        self.assertAlmostEqual(actions[1][3], 1.5707963267948966)
        self.assertEqual(actions[2][:3], [-1.0, 0.0, -1.0])
        self.assertAlmostEqual(abs(actions[2][3]), 3.141592653589793)

    def test_translation_heading_uses_body_frame_xy(self):
        self.assertAlmostEqual(heading_delta_from_translation(1, 1), 0.7853981633974483)
        self.assertEqual(heading_delta_from_translation(0, 0), 0.0)

    def test_pi0_openfly_actions_reproduce_legacy_heading_and_clamp_stop(self):
        actions = pi0_actions_to_canonical(
            [
                [2.0, 0.0, -0.5, -0.1],
                [1.0, 1.0, 0.0, 1.2],
            ]
        )
        self.assertEqual(actions[0], [2.0, 0.0, -0.5, 0.0, 0.0])
        self.assertAlmostEqual(actions[1][3], 0.7853981633974483)
        self.assertEqual(actions[1][4], 1.0)

    def test_pi0_action_conversion_rejects_bad_shape_and_nonfinite_values(self):
        for rows in ([[1, 2, 3]], [[1, 2, float("nan"), 0]]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                pi0_actions_to_canonical(rows)

    def test_omninav_prompt_matches_openfly_training_format(self):
        prompt = build_omninav_prompt("  fly to the red roof  ")
        self.assertIn(
            "# Historical front-camera observations and UAV positions: "
            "<input_pos1><image> <input_pos2><image> <input_pos3><image> "
            "<input_pos4><image> <input_pos5><image>\n",
            prompt,
        )
        self.assertIn("# Current front-camera observation and UAV position: <input_target><image>\n", prompt)
        self.assertIn("# Mission: fly to the red roof <|NAV|>\n", prompt)

    def test_omninav_history_selection_matches_training_policy(self):
        self.assertEqual(select_history_indices(0), [0, 0, 0, 0, 0])
        self.assertEqual(select_history_indices(3), [0, 0, 0, 1, 2])
        self.assertEqual(select_history_indices(9), [0, 2, 4, 6, 8])

    def test_omninav_history_positions_use_current_body_frame(self):
        positions = history_positions_in_current_body(
            [[1, 2, 3, 0], [2, 1, 2, 0]],
            [1, 1, 1, 1.5707963267948966],
        )
        self.assertAlmostEqual(positions[0][0], 1.0)
        self.assertAlmostEqual(positions[0][1], 0.0, places=7)
        self.assertEqual(positions[0][2], 2.0)
        self.assertAlmostEqual(positions[1][0], 0.0, places=7)
        self.assertAlmostEqual(positions[1][1], -1.0)

    def test_omninav_cumulative_waypoints_become_canonical_chunk(self):
        actions = omninav_waypoints_to_actions(
            [[1, 0, 0], [1, 2, -1], [0, 2, -1]],
            [0.1, 0.2, 0.9],
        )
        self.assertEqual(actions[0], [1.0, 0.0, 0.0, 0.0, 0.1])
        self.assertEqual(actions[1][:3], [0.0, 2.0, -1.0])
        self.assertAlmostEqual(actions[1][3], 1.5707963267948966)
        self.assertEqual(actions[2][:3], [-1.0, 0.0, 0.0])
        self.assertAlmostEqual(abs(actions[2][3]), 3.141592653589793)
        self.assertEqual(actions[2][4], 0.9)

    def test_predict_request_normalizes_legacy_and_named_views(self):
        parsed = PredictRequest.from_mapping(
            {
                "episode_id": "ep",
                "state": [0, 0, 0, 0],
                "pose": [0, 0, 0, 0],
                "image_base64": "front-data",
                "images_base64": {"down": "down-data"},
                "primary_view": "front",
                "auxiliary_state": {"speed": 2},
                "camera_specs": {"front": {"width": 640, "height": 480}},
            }
        )
        self.assertEqual(parsed.available_views, ("down", "front"))
        self.assertEqual(parsed.images_base64["front"], "front-data")
        self.assertEqual(parsed.auxiliary_state["speed"], 2)
        self.assertEqual(parsed.camera_specs["front"]["width"], 640)

    def test_function_backend_maps_protocol_fields_to_inference_kwargs(self):
        backend = FunctionBackend(
            entrypoint="test_model_server:function_model_fixture",
            inputs={"prompt": "instruction", "position": "pose"},
            static_kwargs={"scale": 2},
        )
        parsed = PredictRequest.from_mapping({"episode_id": "ep", "instruction": "go", "pose": [1, 2, 3, 0]})
        self.assertEqual(backend.predict(parsed)["actions"], [[2, 4, 6, 0, 0]])


if __name__ == "__main__":
    unittest.main()


def function_model_fixture(prompt, position, scale):
    assert prompt == "go"
    return [[position[0] * scale, position[1] * scale, position[2] * scale, 0, 0]]
