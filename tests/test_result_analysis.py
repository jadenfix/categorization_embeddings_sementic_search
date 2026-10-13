"""Regression coverage for truthful classification-result semantics."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from cli.analyze_results import ResultsAnalyzer

ROOT = Path(__file__).resolve().parents[1]


def test_execution_classification_abstention_and_failure_are_distinct():
    rows = pd.DataFrame(
        [
            {
                "name": "classified",
                "category_name": "Cafe",
                "category_label": "Cafe",
                "similarity_score": 0.8,
                "status": "success",
            },
            {
                "name": "python-none",
                "category_name": None,
                "category_label": None,
                "similarity_score": 0.0,
                "status": "success",
            },
            {
                "name": "none",
                "category_name": "None",
                "category_label": "None",
                "similarity_score": 0.0,
                "status": "success",
            },
            {
                "name": "unknown",
                "category_name": "Unknown",
                "category_label": "Unknown",
                "similarity_score": 0.0,
                "status": "success",
            },
            {
                "name": "null",
                "category_name": "null",
                "category_label": "NULL",
                "similarity_score": 0.0,
                "status": "success",
            },
            {
                "name": "blank",
                "category_name": "",
                "category_label": "   ",
                "similarity_score": 0.0,
                "status": "success",
            },
            {
                "name": "failed",
                "category_name": "Cafe",
                "category_label": "Cafe",
                "similarity_score": 0.9,
                "status": "failed",
            },
        ]
    )

    analyzer = ResultsAnalyzer(rows)
    stats = analyzer.calculate_basic_statistics()

    assert stats == {
        "total_venues": 7,
        "execution_succeeded": 6,
        "classified_venues": 1,
        "abstained_venues": 5,
        "failed_venues": 1,
        "execution_success_rate": pytest.approx(600 / 7),
        "classification_coverage": pytest.approx(100 / 7),
        "avg_similarity_score": 0.8,
        "median_similarity_score": 0.8,
        "min_similarity_score": 0.8,
        "max_similarity_score": 0.8,
        "std_similarity_score": pytest.approx(float("nan"), nan_ok=True),
    }
    assert analyzer.analyze_category_distribution()["most_common_categories"] == [
        ("Cafe", 1)
    ]
    assert analyzer.analyze_similarity_distribution()["high_confidence_count"] == 1


def test_checked_in_34_row_historical_output_has_truthful_coverage():
    results = pd.read_csv(ROOT / "results" / "venue_category_matches.csv")
    analyzer = ResultsAnalyzer(results)
    stats = analyzer.calculate_basic_statistics()

    assert stats["total_venues"] == 34
    assert stats["execution_succeeded"] == 34
    assert stats["classified_venues"] == 17
    assert stats["abstained_venues"] == 17
    assert stats["failed_venues"] == 0
    assert stats["execution_success_rate"] == 100.0
    assert stats["classification_coverage"] == 50.0
    assert (
        sum(
            analyzer.analyze_category_distribution()[
                "hierarchy_level_distribution"
            ].values()
        )
        == 17
    )
    assert (
        sum(
            analyzer.analyze_similarity_distribution()[
                "similarity_distribution"
            ].values()
        )
        == 17
    )


def test_provider_pipeline_exits_nonzero_without_credentials(tmp_path):
    environment = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYTHONPATH": str(ROOT),
        "VENUE_PROJECT_ROOT": str(tmp_path),
    }
    completed = subprocess.run(
        [sys.executable, str(ROOT / "process_all_venues.py")],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "OPENAI_API_KEY environment variable is required" in (
        completed.stdout + completed.stderr
    )
