"""Configuration portability without external API calls."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def load(self, environment):
        spec = importlib.util.spec_from_file_location("test_venue_config", ROOT / "config.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict("os.environ", environment, clear=True), patch("dotenv.load_dotenv"):
            spec.loader.exec_module(module)
        return module

    def test_standard_api_keys_and_checkout_relative_paths(self):
        config = self.load({"OPENAI_API_KEY": "fake-openai", "SERPER_API_KEY": "fake-serper"})
        config.validate_config()
        self.assertEqual(config.OPENAI_API_KEY, "fake-openai")
        self.assertEqual(config.SERPER_API_KEY, "fake-serper")
        self.assertEqual(Path(config.DATA_DIR), ROOT / "data")
        self.assertEqual(Path(config.RESULTS_DIR), ROOT / "results")

    def test_legacy_key_names_are_still_accepted(self):
        config = self.load({"OPEN_AI_API_KEY": "legacy-openai", "SERPER_KEY": "legacy-serper"})
        config.validate_config()
        self.assertEqual(config.OPENAI_API_KEY, "legacy-openai")
        self.assertEqual(config.SERPER_API_KEY, "legacy-serper")

    def test_dataset_root_override(self):
        config = self.load({"VENUE_PROJECT_ROOT": "/tmp/venue-custom"})
        self.assertEqual(Path(config.DATA_DIR), Path("/tmp/venue-custom/data"))

    def test_missing_keys_fail_before_api_work(self):
        with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
            self.load({}).validate_config()
