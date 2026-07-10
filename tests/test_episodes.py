import json
import tempfile
import unittest
from pathlib import Path

from uav_eval.episodes import load_airbrain_episodes


class EpisodeLoaderTest(unittest.TestCase):
    def test_airbrain_schema(self):
        with tempfile.TemporaryDirectory() as root:
            episode_dir = Path(root) / "env_airsim_test" / "traj_1" / "episode_0"
            episode_dir.mkdir(parents=True)
            pose = {
                "actions": [
                    {"pos": [0, 0, 1], "yaw": 0, "bbox_info": [{"id": 7}]},
                    {"pos": [3, 0, 1], "yaw": 0},
                ],
                "aim_landmark_0": {"position": [3, 0, 1]},
            }
            (episode_dir / "pose_bbox_updated.json").write_text(json.dumps(pose), encoding="utf-8")
            config = Path(root) / "eval.json"
            config.write_text(
                json.dumps([{"path": str(episode_dir), "vla_caption": {"result": "go"}}]), encoding="utf-8"
            )
            episodes = load_airbrain_episodes(config)
            self.assertEqual(len(episodes), 1)
            self.assertEqual(episodes[0].env_name, "env_airsim_test")
            self.assertEqual(episodes[0].target_id, 7)
            self.assertAlmostEqual(episodes[0].reference_path_length, 3)


if __name__ == "__main__":
    unittest.main()
