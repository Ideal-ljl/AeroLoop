import tempfile
import unittest
from pathlib import Path

from uav_eval.cli import main
from uav_eval.config import load_config


class CliTest(unittest.TestCase):
    def test_init_config_exports_packaged_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            main(["init-config", "--template", "mock", "--output", str(path)])
            config = load_config(path)
            self.assertEqual(config["benchmark"]["source"], "inline")
            self.assertFalse(config["media"]["save_video"])


if __name__ == "__main__":
    unittest.main()
