"""Tests for EDA notebook export utilities."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from notebooks import export_eda


def _mock_hypotheses() -> list[str]:
    """Return deterministic hypotheses for export tests."""
    return [
        "Hypothesis 1 about thematic versus off-topic rows.",
        "Hypothesis 2 about HTML cleanup needs.",
        "Hypothesis 3 about short text handling.",
        "Hypothesis 4 about source-aware annotation.",
        "Hypothesis 5 about class separation quality.",
    ]


def test_export_eda_runs_without_error(monkeypatch) -> None:
    """EDA export should run and create a sizable HTML report."""
    monkeypatch.setattr(
        export_eda.GeminiLLMClient,
        "generate_stopwords",
        lambda self, topic, base_stopwords: set(base_stopwords),
    )
    monkeypatch.setattr(
        export_eda.GeminiLLMClient,
        "generate_eda_hypotheses",
        lambda self, dataset_summary: _mock_hypotheses(),
    )

    report_path = export_eda.export_eda_report(ROOT, ROOT / "reports" / "eda_report.html")

    assert report_path.exists()
    assert report_path.stat().st_size > 10_000


def test_wordcloud_files_created(monkeypatch) -> None:
    """WordCloud export should create both expected PNG files."""
    monkeypatch.setattr(
        export_eda.GeminiLLMClient,
        "generate_stopwords",
        lambda self, topic, base_stopwords: set(base_stopwords),
    )
    monkeypatch.setattr(
        export_eda.GeminiLLMClient,
        "generate_eda_hypotheses",
        lambda self, dataset_summary: _mock_hypotheses(),
    )

    export_eda.export_eda_report(ROOT, ROOT / "reports" / "eda_report.html")

    assert (ROOT / "reports" / "wordcloud_all.png").exists()
    assert (ROOT / "reports" / "wordcloud_domain.png").exists()


def test_hypotheses_json_created(monkeypatch) -> None:
    """EDA export should persist hypotheses JSON locally."""
    monkeypatch.setattr(
        export_eda.GeminiLLMClient,
        "generate_stopwords",
        lambda self, topic, base_stopwords: set(base_stopwords),
    )
    monkeypatch.setattr(
        export_eda.GeminiLLMClient,
        "generate_eda_hypotheses",
        lambda self, dataset_summary: _mock_hypotheses(),
    )

    export_eda.export_eda_report(ROOT, ROOT / "reports" / "eda_report.html")

    hypotheses_path = ROOT / "reports" / "eda_hypotheses.json"
    assert hypotheses_path.exists()
    payload = json.loads(hypotheses_path.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert len(payload) >= 3


def test_eda_metadata_json_created(monkeypatch) -> None:
    """EDA export should persist metadata describing the topic and input mtimes."""
    monkeypatch.setattr(
        export_eda.GeminiLLMClient,
        "generate_stopwords",
        lambda self, topic, base_stopwords: set(base_stopwords),
    )
    monkeypatch.setattr(
        export_eda.GeminiLLMClient,
        "generate_eda_hypotheses",
        lambda self, dataset_summary: _mock_hypotheses(),
    )

    export_eda.export_eda_report(ROOT, ROOT / "reports" / "eda_report.html")

    metadata_path = ROOT / "reports" / "eda_metadata.json"
    assert metadata_path.exists()
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert "topic" in payload
    assert "raw_dataset_mtime" in payload


def test_build_insights_stays_topic_aware() -> None:
    """Generic EDA insights should not hardcode sailing-specific wording for other topics."""
    df = pd.DataFrame(
        {
            "text": [
                "cars engine tuning speed track braking",
                "cars maintenance garage diagnostics review",
                "general emotion text unrelated",
            ],
            "source": [
                "topic_bootstrap_guides",
                "topic_bootstrap_forum",
                "huggingface_demo",
            ],
            "text_len": [39, 41, 29],
        }
    )
    source_df = export_eda.source_distribution_frame(df)
    thematic_stats = export_eda.compute_thematic_stats(df)
    quality_df = pd.DataFrame(
        [
            {
                "source": "topic_bootstrap_guides",
                "html_entities": 0.0,
                "short_texts(<50)": 10.0,
                "long_texts(>1000)": 0.0,
                "duplicates_pct": 0.0,
            },
            {
                "source": "huggingface_demo",
                "html_entities": 5.0,
                "short_texts(<50)": 50.0,
                "long_texts(>1000)": 0.0,
                "duplicates_pct": 0.0,
            },
        ]
    )

    insights = export_eda.build_insights(
        df,
        source_df,
        thematic_stats,
        quality_df,
        stopwords=set(),
        topic="cars",
    )

    combined = " ".join(insights.values()).lower()
    assert "sailingforums" not in combined
    assert "yachtingworld" not in combined
    assert "cars" in combined


def test_load_inputs_prefers_clean_dataset(tmp_path: Path) -> None:
    """EDA should use dataset_clean.parquet when it exists."""
    raw_dir = tmp_path / "data" / "raw"
    reports_dir = tmp_path / "reports"
    raw_dir.mkdir(parents=True)
    reports_dir.mkdir(parents=True)

    raw_df = pd.DataFrame(
        {
            "text": ["raw topic text"],
            "source": ["raw_source"],
        }
    )
    clean_df = pd.DataFrame(
        {
            "text": ["clean topic text"],
            "source": ["clean_source"],
        }
    )
    raw_df.to_parquet(raw_dir / "dataset.parquet", index=False)
    clean_df.to_parquet(raw_dir / "dataset_clean.parquet", index=False)
    (reports_dir / "domain_spec.json").write_text(
        json.dumps({"keywords_by_class": {}}),
        encoding="utf-8",
    )

    loaded_df, _ = export_eda.load_inputs(tmp_path)

    assert loaded_df.iloc[0]["text"] == "clean topic text"
    assert loaded_df.iloc[0]["source"] == "clean_source"


def test_build_stopwords_filters_topic_and_fallback_class_tokens(monkeypatch, tmp_path: Path) -> None:
    """Topic tokens and fallback class suffixes should be removed from WordCloud vocabulary."""
    (tmp_path / "reports").mkdir(parents=True)
    config = {
        "domain": {
            "topic": "fitness",
            "classes": [
                "fitness_basics",
                "fitness_tools",
                "fitness_workflows",
                "fitness_issues",
                "fitness_advanced",
            ],
        }
    }
    (tmp_path / "config.yaml").write_text(
        json.dumps(config),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        export_eda.GeminiLLMClient,
        "generate_stopwords",
        lambda self, topic, base_stopwords: set(base_stopwords),
    )

    stopwords = export_eda.build_stopwords(
        {"keywords_by_class": {}},
        topic="fitness",
        project_root=tmp_path,
    )

    assert "fitness" in stopwords
    assert "basics" in stopwords
    assert "tools" in stopwords
    assert "workflows" in stopwords
    assert "issues" in stopwords
    assert "advanced" in stopwords
