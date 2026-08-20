import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aeroloop.cli import main
from aeroloop.config import load_config


class CliTest(unittest.TestCase):
    def test_init_config_exports_packaged_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            main(["init-config", "--template", "mock", "--output", str(path)])
            config = load_config(path)
            self.assertEqual(config["benchmark"]["source"], "inline")
            self.assertFalse(config["media"]["save_video"])

    def test_config_inheritance_deep_merges(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "base.yaml").write_text(
                "policy:\n  type: http\n  kwargs:\n    timeout_s: 10\n    url: old\n", encoding="utf-8"
            )
            (root / "child.yaml").write_text(
                "extends: base.yaml\npolicy:\n  kwargs:\n    url: new\noutput:\n  jsonl: out.jsonl\n", encoding="utf-8"
            )
            config = load_config(root / "child.yaml")
            self.assertEqual(config["policy"]["kwargs"]["url"], "new")
            self.assertEqual(config["policy"]["kwargs"]["timeout_s"], 10)

    def test_run_cli_overrides_output_and_media(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(
                "benchmark:\n  source: inline\n"
                "output:\n  jsonl: old.jsonl\nmedia:\n  show_window: true\n  save_video: true\n",
                encoding="utf-8",
            )
            with patch("aeroloop.cli.run") as run_mock:
                main(
                    [
                        "run",
                        "--config",
                        str(path),
                        "--output-jsonl",
                        "new.jsonl",
                        "--headless",
                        "--no-video",
                    ]
                )
            config = run_mock.call_args.args[0]
            self.assertEqual(config["output"]["jsonl"], "new.jsonl")
            self.assertFalse(config["media"]["show_window"])
            self.assertFalse(config["media"]["save_video"])

    def test_init_http_config_uses_simulator_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            main(["init-config", "--template", "http", "--output", str(path)])
            config = load_config(path)
            self.assertIn("simulator", config)
            self.assertEqual(config["policy"]["type"], "http")


if __name__ == "__main__":
    unittest.main()
