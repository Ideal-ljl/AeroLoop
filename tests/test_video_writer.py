import importlib.util
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from uav_eval.media import FFmpegVideoWriter
from uav_eval.envs.mock import MockEnvironment
from uav_eval.media import MediaConfig, MediaObserver
from uav_eval.policies.mock import MockPolicy
from uav_eval.runner import RolloutConfig, RolloutRunner
from uav_eval.types import EpisodeSpec, Pose


@unittest.skipUnless(
    importlib.util.find_spec("numpy") is not None
    and importlib.util.find_spec("cv2") is not None
    and shutil.which("ffmpeg") is not None,
    "NumPy, OpenCV, and ffmpeg are required",
)
class VideoWriterTest(unittest.TestCase):
    def test_streams_odd_sized_rgb_frames_to_mp4(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "video.mp4"
            writer = FFmpegVideoWriter(path, fps=5)
            for index in range(5):
                frame = np.zeros((65, 97, 3), dtype=np.uint8)
                frame[..., index % 3] = 40 + index * 30
                writer.write(frame)
            writer.close()
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 500)

    def test_media_observer_records_episode_artifact(self):
        import numpy as np

        class ImageEnvironment(MockEnvironment):
            def _observation(self):
                return replace(super()._observation(), rgb=np.full((64, 96, 3), 80, dtype=np.uint8))

        with tempfile.TemporaryDirectory() as tmp:
            observer = MediaObserver(
                MediaConfig(save_video=True, video_dir=tmp, video_fps=5, save_collision_frame=False)
            )
            episode = EpisodeSpec("video-ep", "mock", "go", Pose(0, 0, 0, 0), (2, 0, 0), 2)
            result = RolloutRunner(
                ImageEnvironment(),
                MockPolicy(action=(1, 0, 0, 0, 0)),
                RolloutConfig(max_steps=2),
                observers=[observer],
            ).run_episode(episode)
            observer.close()
            video_path = Path(result.artifacts["video_path"])
            self.assertTrue(video_path.is_file())
            self.assertGreater(video_path.stat().st_size, 500)


if __name__ == "__main__":
    unittest.main()
