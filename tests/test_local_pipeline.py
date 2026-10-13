"""Local file and candidate-matching checks; no external service is called."""
import importlib
from pathlib import Path
import numpy as np
import pytest
import config
from core.category_matcher import CategoryMatcher
from utils.data_loader import DataLoader


def test_default_paths_find_bundled_venue_and_taxonomy_files():
    root = Path(config.__file__).resolve().parent
    assert Path(config.PROJECT_ROOT) == root
    categories = DataLoader.load_csv_data(config.CATEGORIES_FILE)
    venues = DataLoader.load_csv_data(config.VENUES_FILE)
    assert not categories.empty and not venues.empty


def test_standard_openai_key_precedes_legacy_alias(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "standard-test-key")
    monkeypatch.setenv("OPEN_AI_API_KEY", "legacy-test-key")
    try:
        importlib.reload(config)
        assert config.OPENAI_API_KEY == "standard-test-key"
        monkeypatch.delenv("OPENAI_API_KEY")
        importlib.reload(config)
        assert config.OPENAI_API_KEY == "legacy-test-key"
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_local_embedding_candidates_rank_without_service_calls():
    metadata = {"categories": [{"Category Name": "Coffee Shop", "Category Label": "cafe"},
                                {"Category Name": "Hospital", "Category Label": "hospital"}]}
    matcher = CategoryMatcher(np.array([[1., 0.], [0., 1.]]), metadata)
    assert matcher.find_best_category_match([1., 0.]) == ("Coffee Shop", "cafe", 1.)
    assert matcher.find_top_k_matches([0., 1.], 1) == [("Hospital", "hospital", 1.)]
