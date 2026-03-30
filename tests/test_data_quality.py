"""Tests for DataQualityAgent."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from agents.data_quality_agent import DataQualityAgent


def _write_config(base_dir: Path) -> Path:
    """Create a minimal config file for data quality tests."""
    config = {
        "project": {
            "name": "smart-data-pipeline",
            "version": "0.1.0",
            "description": "Test config",
        },
        "domain": {
            "topic": "sailing and yacht navigation",
            "classes": ["navigation", "safety", "equipment", "weather", "licensing"],
            "review_label": "other_or_offtopic",
            "language": "en",
        },
        "sources": {},
        "data": {
            "raw_path": "data/raw",
            "labeled_path": "data/labeled",
            "review_queue_path": "data/review_queue.csv",
        },
        "annotation": {"confidence_threshold": 0.7, "model": "zero-shot"},
        "active_learning": {
            "strategy": "entropy",
            "initial_size": 50,
            "batch_size": 20,
            "n_iterations": 5,
        },
        "model": {"type": "sklearn", "base_model": "logistic_regression"},
        "llm": {
            "provider": "gemini",
            "model": "models/gemini-flash-latest",
            "fallback_models": [
                "models/gemini-2.5-flash",
                "models/gemini-flash-latest",
                "models/gemma-3-4b-it",
            ],
            "max_tokens": 1000,
            "temperature": 0.3,
            "max_prompt_chars": 2000,
        },
        "pipeline": {"orchestrator": "prefect", "save_intermediate": True},
        "quality": {
            "strategy": {
                "html_entities": "decode",
                "html_artifacts": "remove",
                "duplicates": "drop",
                "fuzzy_duplicates": "remove",
                "short_texts": "filter",
                "long_texts": "truncate",
                "missing": "drop",
            },
            "thresholds": {"short_text_min": 50, "long_text_max": 1000},
            "fuzzy_threshold": 90.0,
            "html_artifact_patterns": [
                "wp",
                "timeincuk",
                "inspirewp",
                "keyassets",
                "figcaption",
                "webfeedsfeaturedvisual",
                "attachment",
            ],
        },
    }
    config_path = base_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


@pytest.fixture
def temp_project(tmp_path: Path) -> Path:
    """Create an isolated temporary project layout."""
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "labeled").mkdir(parents=True)
    (tmp_path / "reports").mkdir(parents=True)
    _write_config(tmp_path)
    return tmp_path


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Return a sample dataframe with realistic sailing rows."""
    long_text = "storm routing " * 150
    return pd.DataFrame(
        {
            "id": ["1", "2", "3", "4"],
            "text": [
                "This sailing text contains &amp; entity and wp-content attachment-medium artifacts for cleanup in the report.",
                "Short note",
                "This sailing text contains &amp; entity and wp-content attachment-medium artifacts for cleanup in the report.",
                long_text,
            ],
            "label": ["unlabeled", "unlabeled", "unlabeled", "unlabeled"],
            "source": [
                "rss_www.yachtingworld.com",
                "sailingforums",
                "stackexchange_sailing",
                "rss_www.cruisingworld.com",
            ],
            "collected_at": [
                "2026-03-30T10:00:00+00:00",
                "2026-03-30T10:01:00+00:00",
                "2026-03-30T10:02:00+00:00",
                "2026-03-30T10:03:00+00:00",
            ],
        }
    )


def _make_agent(temp_project: Path) -> DataQualityAgent:
    """Build an agent pointing at the temp project config."""
    return DataQualityAgent(config_path=str(temp_project / "config.yaml"))


def test_agent_initializes(temp_project: Path) -> None:
    """Agent should initialize config, thresholds, and llm client."""
    agent = _make_agent(temp_project)
    assert agent._short_text_min == 50
    assert agent._long_text_max == 1000
    assert agent._llm_client is not None
    assert agent.llm is not None


def test_detect_issues_finds_html_entities(temp_project: Path) -> None:
    """HTML entities should be detected in encoded text."""
    agent = _make_agent(temp_project)
    df = pd.DataFrame({"text": ["hello &amp; world &#39;test&#39;"], "label": ["unlabeled"]})

    report = agent.detect_issues(df)

    assert report["html_entities"]["count"] > 0


def test_detect_issues_finds_duplicates(temp_project: Path) -> None:
    """Duplicate texts should be reported."""
    agent = _make_agent(temp_project)
    df = pd.DataFrame(
        {
            "text": ["same sailing text for duplicate detection", "same sailing text for duplicate detection"],
            "label": ["unlabeled", "unlabeled"],
        }
    )

    report = agent.detect_issues(df)

    assert report["duplicates"]["count"] == 1


def test_detect_issues_finds_short_texts(temp_project: Path) -> None:
    """Short texts should be counted with the configured threshold."""
    agent = _make_agent(temp_project)
    df = pd.DataFrame({"text": ["tiny"], "label": ["unlabeled"]})

    report = agent.detect_issues(df)

    assert report["short_texts"]["count"] == 1


def test_fix_decodes_html_entities(temp_project: Path) -> None:
    """HTML entities should be decoded when strategy requests it."""
    agent = _make_agent(temp_project)
    df = pd.DataFrame(
        {
            "text": ["This sailing sentence is long enough &amp; includes &#39;quoted&#39; content for decoding."],
            "label": ["unlabeled"],
        }
    )

    cleaned = agent.fix(
        df,
        {
            "html_entities": "decode",
            "html_artifacts": "keep",
            "duplicates": "keep",
            "short_texts": "keep",
            "long_texts": "keep",
            "missing": "fill",
        },
    )

    assert "&amp;" not in cleaned["text"].iloc[0]
    assert "&#39;" not in cleaned["text"].iloc[0]
    assert "&" in cleaned["text"].iloc[0]
    assert "'" in cleaned["text"].iloc[0]


def test_fix_removes_html_artifacts(temp_project: Path) -> None:
    """Configured HTML artifact tokens should be removed from text."""
    agent = _make_agent(temp_project)
    df = pd.DataFrame(
        {
            "text": ["This sailing article references wp-content attachment-medium figure and still stays informative enough."],
            "label": ["unlabeled"],
        }
    )

    cleaned = agent.fix(
        df,
        {
            "html_entities": "keep",
            "html_artifacts": "remove",
            "duplicates": "keep",
            "short_texts": "keep",
            "long_texts": "keep",
            "missing": "fill",
        },
    )

    assert "wp-content" not in cleaned["text"].iloc[0].lower()
    assert "attachment-medium" not in cleaned["text"].iloc[0].lower()


def test_fix_drops_duplicates(temp_project: Path) -> None:
    """Duplicate rows should be removed when requested."""
    agent = _make_agent(temp_project)
    df = pd.DataFrame(
        {
            "text": ["same sailing duplicate text for cleanup", "same sailing duplicate text for cleanup"],
            "label": ["unlabeled", "unlabeled"],
        }
    )

    cleaned = agent.fix(
        df,
        {
            "html_entities": "keep",
            "html_artifacts": "keep",
            "duplicates": "drop",
            "fuzzy_duplicates": "keep",
            "short_texts": "keep",
            "long_texts": "keep",
            "missing": "fill",
        },
    )

    assert len(cleaned) == 1


def test_fix_filters_short_texts(temp_project: Path) -> None:
    """Short texts should be filtered out below threshold."""
    agent = _make_agent(temp_project)
    df = pd.DataFrame(
        {
            "text": ["tiny", "This sailing discussion is comfortably longer than fifty characters for retention."],
            "label": ["unlabeled", "unlabeled"],
        }
    )

    cleaned = agent.fix(
        df,
        {
            "html_entities": "keep",
            "html_artifacts": "keep",
            "duplicates": "keep",
            "fuzzy_duplicates": "keep",
            "short_texts": "filter",
            "long_texts": "keep",
            "missing": "fill",
        },
    )

    assert len(cleaned) == 1
    assert cleaned["text"].str.len().min() >= 50


def test_fix_truncates_long_texts(temp_project: Path) -> None:
    """Long texts should be truncated to the configured max length."""
    agent = _make_agent(temp_project)
    df = pd.DataFrame(
        {
            "text": ["x" * 2000],
            "label": ["unlabeled"],
        }
    )

    cleaned = agent.fix(
        df,
        {
            "html_entities": "keep",
            "html_artifacts": "keep",
            "duplicates": "keep",
            "fuzzy_duplicates": "keep",
            "short_texts": "keep",
            "long_texts": "truncate",
            "missing": "fill",
        },
    )

    assert len(cleaned["text"].iloc[0]) == 1000


def test_compare_returns_improvements(temp_project: Path, sample_df: pd.DataFrame) -> None:
    """Compare should return before/after counts and improvement keys."""
    agent = _make_agent(temp_project)
    cleaned = agent.fix(sample_df, agent._strategy)

    compare = agent.compare(sample_df, cleaned)

    assert {"rows_before", "rows_after", "improvements"} <= compare.keys()
    assert "duplicates_before" in compare["improvements"]


def test_run_saves_clean_parquet(temp_project: Path, sample_df: pd.DataFrame) -> None:
    """Run should save the cleaned parquet to data/raw/dataset_clean.parquet."""
    agent = _make_agent(temp_project)
    dataset_path = temp_project / "data" / "raw" / "dataset.parquet"
    sample_df.to_parquet(dataset_path, index=False)

    agent.run()

    assert (temp_project / "data" / "raw" / "dataset_clean.parquet").exists()


def test_run_saves_quality_report(temp_project: Path, sample_df: pd.DataFrame) -> None:
    """Run should save a human-readable quality report."""
    agent = _make_agent(temp_project)
    dataset_path = temp_project / "data" / "raw" / "dataset.parquet"
    sample_df.to_parquet(dataset_path, index=False)

    agent.run()

    assert (temp_project / "reports" / "quality_report.md").exists()


def test_context_memory_updated(temp_project: Path, sample_df: pd.DataFrame) -> None:
    """Run should update context memory to step 2.1."""
    agent = _make_agent(temp_project)
    dataset_path = temp_project / "data" / "raw" / "dataset.parquet"
    sample_df.to_parquet(dataset_path, index=False)

    agent.run()

    payload = json.loads((temp_project / "reports" / "context_memory.json").read_text(encoding="utf-8"))
    assert payload["step"] == "2.1"
    assert payload["notes"] == "DataQualityAgent applied cleaning"


def test_summary_no_raw_data(temp_project: Path, sample_df: pd.DataFrame) -> None:
    """Summary should remain compact and exclude raw text payloads."""
    agent = _make_agent(temp_project)
    dataset_path = temp_project / "data" / "raw" / "dataset.parquet"
    sample_df.to_parquet(dataset_path, index=False)

    agent.run()
    summary = agent.summary()
    encoded = json.dumps(summary, ensure_ascii=False)

    assert {"texts", "rows", "data"}.isdisjoint(summary.keys())
    assert "wp-content attachment-medium artifacts" not in encoded


def test_explain_issues_returns_dict(
    temp_project: Path, sample_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LLM advice should return a validated dict when Gemini JSON is mocked."""
    agent = _make_agent(temp_project)
    quality_report = agent.detect_issues(sample_df)
    monkeypatch.setattr(
        agent.llm,
        "generate_json",
        lambda prompt: {
            "summary": "Качество данных среднее, но проблемы хорошо локализованы.",
            "top_issues": ["HTML entities", "Короткие тексты", "HTML artifacts"],
            "recommended_strategy": {
                "html_entities": "decode — это очистит текст",
                "duplicates": "drop — повторы мешают",
                "short_texts": "filter — мало сигнала",
                "long_texts": "truncate — сохраняем смысл",
            },
            "risks": ["Риск потери части данных", "Риск смещения модели"],
            "confidence": "high",
        },
    )

    advice = agent.explain_issues(quality_report)

    assert isinstance(advice, dict)
    assert {"summary", "top_issues", "recommended_strategy"} <= advice.keys()


def test_explain_issues_fallback(
    temp_project: Path, sample_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fallback advice should be returned when Gemini JSON generation fails."""
    agent = _make_agent(temp_project)
    quality_report = agent.detect_issues(sample_df)

    def _raise(_: str) -> dict[str, Any]:
        raise Exception("gemini unavailable")

    monkeypatch.setattr(agent.llm, "generate_json", _raise)

    advice = agent.explain_issues(quality_report)

    assert isinstance(advice, dict)
    assert "summary" in advice
    assert advice["confidence"] == "medium"


def test_explain_issues_saves_json(
    temp_project: Path, sample_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Advice generation should save reports/llm_quality_advice.json."""
    agent = _make_agent(temp_project)
    quality_report = agent.detect_issues(sample_df)
    monkeypatch.setattr(
        agent.llm,
        "generate_json",
        lambda prompt: {
            "summary": "Сводка по качеству данных.",
            "top_issues": ["HTML entities", "Короткие тексты", "Длинные тексты"],
            "recommended_strategy": {
                "html_entities": "decode",
                "duplicates": "drop",
                "short_texts": "filter",
                "long_texts": "truncate",
            },
            "risks": ["Риск потери части данных"],
            "confidence": "medium",
        },
    )

    agent.explain_issues(quality_report)

    assert (temp_project / "reports" / "llm_quality_advice.json").exists()


def test_fuzzy_duplicates_finds_similar(temp_project: Path) -> None:
    """Fuzzy matching should find near-duplicate sailing texts over the threshold."""
    agent = _make_agent(temp_project)
    df = pd.DataFrame(
        {
            "text": [
                "Sailing in bad weather requires careful route planning and safety checks.",
                "Sailing in bad weather requires careful route planning and safety checks!",
                "Anchoring tips for quiet bays and overnight stays.",
            ],
            "label": ["unlabeled", "unlabeled", "unlabeled"],
        }
    )

    result = agent.find_fuzzy_duplicates(df, threshold=90.0)

    assert result["fuzzy_duplicate_pairs"] >= 1
    assert result["examples"]


def test_fix_removes_fuzzy_duplicates(temp_project: Path) -> None:
    """Near-duplicate rows should be removed when fuzzy_duplicates=remove."""
    agent = _make_agent(temp_project)
    df = pd.DataFrame(
        {
            "text": [
                "Sailing in bad weather requires careful route planning and safety checks.",
                "Sailing in bad weather requires careful route planning and safety checks!",
                "Anchoring tips for quiet bays and overnight stays with enough context.",
            ],
            "label": ["unlabeled", "unlabeled", "unlabeled"],
        }
    )

    cleaned = agent.fix(
        df,
        {
            "html_entities": "keep",
            "html_artifacts": "keep",
            "duplicates": "keep",
            "fuzzy_duplicates": "remove",
            "short_texts": "keep",
            "long_texts": "keep",
            "missing": "fill",
        },
    )

    assert len(cleaned) == 2
