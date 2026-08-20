import unittest
import json
import tempfile
from pathlib import Path

from aeroloop.datasets import load_openfly_episodes, load_traveluav_episodes, write_dataset_html
from aeroloop.episodes import load_inline_episodes


class EpisodeLoaderTest(unittest.TestCase):
    def test_inline_schema(self):
        episodes = load_inline_episodes(
            [
                {
                    "episode_id": "sim:1",
                    "env_name": "custom-sim",
                    "instruction": "go",
                    "start_pose": [0, 0, 1, 0],
                    "target_position": [3, 0, 1],
                    "metadata": {"scene_target_id": 7},
                }
            ]
        )
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0].env_name, "custom-sim")
        self.assertEqual(episodes[0].metadata["scene_target_id"], 7)
        self.assertAlmostEqual(episodes[0].reference_path_length, 3)

    def test_openfly_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trajectory = root / "env_demo" / "traj_1"
            trajectory.mkdir(parents=True)
            (trajectory / "pose_bbox_updated.json").write_text(
                json.dumps({"actions": [{"pos": [0, 0, 1], "yaw": 0}, {"pos": [3, 4, 1], "yaw": 1}]}),
                encoding="utf-8",
            )
            split = root / "openfly.json"
            split.write_text(
                json.dumps(
                    [
                        {
                            "path": "env_demo/traj_1",
                            "vla_caption": {"result": "go"},
                            "vln_data": [{"target_caption": "the final red tower"}],
                        }
                    ]
                )
            )
            episodes = load_openfly_episodes(str(split), dataset_root=str(root))
            self.assertEqual(episodes[0].env_name, "env_demo")
            self.assertEqual(episodes[0].instruction, "go")
            self.assertEqual(episodes[0].metadata["grounding_prompt"], "the final red tower")
            self.assertAlmostEqual(episodes[0].reference_path_length, 5)

    def test_openfly_adapter_filters_environment_before_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected = root / "env_selected" / "traj_1"
            selected.mkdir(parents=True)
            (selected / "pose_bbox_updated.json").write_text(
                json.dumps({"actions": [{"pos": [1, 2, 3], "yaw": 0}, {"pos": [4, 2, 3], "yaw": 0}]}),
                encoding="utf-8",
            )
            split = root / "openfly.json"
            split.write_text(
                json.dumps(
                    [
                        {"path": "env_excluded/missing", "instruction": "skip"},
                        {"path": "env_selected/traj_1", "instruction": "keep"},
                    ]
                ),
                encoding="utf-8",
            )
            episodes = load_openfly_episodes(
                str(split), dataset_root=str(root), include_envs="env_selected", limit=1
            )
            self.assertEqual(len(episodes), 1)
            self.assertEqual(episodes[0].env_name, "env_selected")
            self.assertEqual(episodes[0].instruction, "keep")
            self.assertEqual(episodes[0].metadata["source_row"], 1)

    def test_traveluav_adapter_deduplicates_frames_and_visualizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trajectory = root / "Map" / "seq"
            trajectory.mkdir(parents=True)
            merged = trajectory / "merged_data.json"
            merged.write_text(
                json.dumps(
                    {
                        "trajectory_raw_detailed": [
                            {"position": [0, 0, 0], "orientation": [0, 0, 0, 1]},
                            {"position": [0, 3, 4], "orientation": [0, 0, 0, 1]},
                        ],
                        "conversations": [{"value": "<image>\nfind it"}],
                    }
                ),
                encoding="utf-8",
            )
            split = root / "split.json"
            split.write_text(
                json.dumps(
                    [
                        {"json": "Map/seq/merged_data.json", "frame": 1},
                        {"json": "Map/seq/merged_data.json", "frame": 2},
                    ]
                ),
                encoding="utf-8",
            )
            episodes = load_traveluav_episodes(str(split), dataset_root=str(root))
            self.assertEqual(len(episodes), 1)
            self.assertEqual(episodes[0].instruction, "find it")
            output = write_dataset_html(episodes, root / "preview.html")
            self.assertIn("find it", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
