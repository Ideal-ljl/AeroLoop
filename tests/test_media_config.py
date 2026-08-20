import unittest

from aeroloop.media import MediaConfig, _safe_name


class MediaConfigTest(unittest.TestCase):
    def test_media_is_disabled_by_default(self):
        config = MediaConfig()
        self.assertFalse(config.show_window)
        self.assertFalse(config.save_video)
        self.assertFalse(config.save_collision_frame)

    def test_episode_filename_is_sanitized(self):
        self.assertEqual(_safe_name("env:test / 01"), "env_test_01")


if __name__ == "__main__":
    unittest.main()
