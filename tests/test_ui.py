"""Tests for Streamlit report generation helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from ui import report_generator


def _write_config(base_dir: Path) -> None:
    """Create a minimal config for UI report tests."""
    config = {
        "domain": {
            "topic": "sailing and yacht navigation",
            "classes": ["navigation", "safety", "equipment", "weather", "licensing"],
            "review_label": "other_or_offtopic",
        },
        "data": {
            "raw_path": "data/raw",
            "labeled_path": "data/labeled",
            "review_queue_path": "data/review_queue.csv",
        },
    }
    (base_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )


@pytest.fixture
def temp_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a small isolated project layout for UI helper tests."""
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "labeled").mkdir(parents=True)
    (tmp_path / "reports").mkdir(parents=True)
    _write_config(tmp_path)

    raw_df = pd.DataFrame(
        {
            "id": ["1", "2"],
            "text": ["navigation text", "safety text"],
            "source": ["stackexchange_sailing", "rss_www.yachtingworld.com"],
        }
    )
    clean_df = raw_df.copy()
    annotated_df = raw_df.copy()
    annotated_df["label"] = ["navigation", "safety"]
    annotated_df["confidence"] = [0.91, 0.42]
    annotated_df["label_source"] = ["zero_shot", "zero_shot"]
    review_df = pd.DataFrame(
        {
            "id": ["2"],
            "text": ["safety text"],
            "label": ["safety"],
            "confidence": [0.42],
            "source": ["rss_www.yachtingworld.com"],
            "suggested_label": ["safety"],
            "corrected_label": [""],
        }
    )

    raw_df.to_parquet(tmp_path / "data" / "raw" / "dataset.parquet", index=False)
    clean_df.to_parquet(tmp_path / "data" / "raw" / "dataset_clean.parquet", index=False)
    annotated_df.to_parquet(tmp_path / "data" / "labeled" / "annotated.parquet", index=False)
    review_df.to_csv(tmp_path / "data" / "review_queue.csv", index=False)
    (tmp_path / "reports" / "context_memory.json").write_text(
        json.dumps({"step": "3.1", "history": {"1.4": {}, "2.1": {}, "3.1": {}}}),
        encoding="utf-8",
    )
    (tmp_path / "reports" / "domain_spec.json").write_text(
        json.dumps({"recommended_classes": ["navigation", "safety"]}),
        encoding="utf-8",
    )
    (tmp_path / "reports" / "quality_report.md").write_text(
        "# Quality\nУдалено строк: 1",
        encoding="utf-8",
    )
    (tmp_path / "reports" / "eda_hypotheses.json").write_text(
        json.dumps(["Гипотеза 1", "Гипотеза 2", "Гипотеза 3"]),
        encoding="utf-8",
    )

    monkeypatch.setattr(report_generator, "PROJECT_ROOT", tmp_path)
    return tmp_path


def test_report_generator_html() -> None:
    """HTML report generation should return a valid HTML document string."""
    sections = {"domain": True, "sources": True}
    data = {
        "topic": "sailing and yacht navigation",
        "steps_completed": ["1.4", "2.1", "3.1"],
        "classes": ["navigation", "safety"],
        "source_distribution": {"stackexchange_sailing": 10},
    }

    content = report_generator.generate_html_report(sections, data)

    assert "<html>" in content.lower()


def test_report_generator_markdown() -> None:
    """Markdown report generation should start with a first-level heading."""
    sections = {"domain": True}
    data = {
        "topic": "sailing and yacht navigation",
        "steps_completed": ["1.4", "2.1", "3.1"],
        "classes": ["navigation", "safety"],
    }

    content = report_generator.generate_markdown_report(sections, data)

    assert content.startswith("#")


def test_telegram_send_fails_gracefully(monkeypatch: pytest.MonkeyPatch) -> None:
    """Telegram sending should return False on a failed API response."""
    class _Response:
        status_code = 401
        text = "unauthorized"

    monkeypatch.setattr(report_generator.requests, "post", lambda *args, **kwargs: _Response())

    result = report_generator.send_telegram_report("Краткая сводка", "bad-token", "123")

    assert result is False


def test_collect_report_data_keys(temp_project: Path) -> None:
    """Collected report data should expose the compact keys used by the UI."""
    data = report_generator.collect_report_data()

    assert isinstance(data, dict)
    assert {"topic", "total_rows", "classes", "steps_completed"} <= data.keys()
    assert data["topic"] == "sailing and yacht navigation"
