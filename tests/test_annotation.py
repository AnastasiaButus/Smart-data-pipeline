"""Tests for AnnotationAgent."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from agents.annotation_agent import AnnotationAgent


def _write_config(base_dir: Path) -> Path:
    """Create a minimal config file for annotation tests."""
    config = {
        "project": {
            "name": "smart-data-pipeline",
            "version": "0.1.0",
            "description": "Test config",
        },
        "domain": {
            "topic": "sailing and yacht navigation",
            "classes": [
                "navigation",
                "safety",
                "equipment",
                "weather",
                "licensing",
            ],
            "review_label": "other_or_offtopic",
            "language": "en",
        },
        "data": {
            "raw_path": "data/raw",
            "labeled_path": "data/labeled",
            "review_queue_path": "data/review_queue.csv",
        },
        "annotation": {
            "confidence_threshold": 0.7,
            "model": "facebook/bart-large-mnli",
            "batch_size": 16,
            "modality": "text",
            "label_source": "zero_shot",
        },
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
    }
    config_path = base_dir / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
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
    """Return a small dataframe for annotation tests."""
    return pd.DataFrame(
        {
            "id": ["1", "2", "3", "4"],
            "text": [
                "Route planning with waypoints, tides, and charts improves coastal navigation decisions.",
                "Life jacket checks and MOB drills improve offshore safety routines for the crew.",
                "Engine maintenance, anchor setup, and rig inspection belong to equipment workflows.",
                "A generic emotional post without domain clues should be routed away from sailing labels.",
            ],
            "source": [
                "stackexchange_sailing",
                "rss_www.yachtingworld.com",
                "sailingforums",
                "huggingface_dair-ai_emotion",
            ],
        }
    )


def _make_agent(temp_project: Path) -> AnnotationAgent:
    """Build an annotation agent pointing to the temp config."""
    return AnnotationAgent(config_path=str(temp_project / "config.yaml"))


def test_agent_initializes(temp_project: Path) -> None:
    """Agent should initialize without loading the zero-shot model eagerly."""
    agent = _make_agent(temp_project)

    assert agent._pipeline is None
    assert agent.review_label == "other_or_offtopic"
    assert agent.classes[:2] == ["navigation", "safety"]
    assert agent._llm_client is not None


def test_auto_label_schema(
    temp_project: Path, sample_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Auto labeling should append label, confidence, and label_source columns."""
    agent = _make_agent(temp_project)

    def _fake_pipeline(
        texts: list[str],
        candidate_labels: list[str],
        multi_label: bool,
    ) -> list[dict[str, object]]:
        assert multi_label is False
        assert "other_or_offtopic" in candidate_labels
        return [
            {
                "labels": [candidate_labels[index % len(candidate_labels)]],
                "scores": [0.91 - (index * 0.05)],
            }
            for index, _ in enumerate(texts)
        ]

    monkeypatch.setattr(agent, "_get_pipeline", lambda: _fake_pipeline)

    labeled = agent.auto_label(sample_df)

    assert {"label", "confidence", "label_source"} <= set(labeled.columns)
    assert labeled["label_source"].eq("zero_shot").all()


def test_auto_label_fallback(
    temp_project: Path, sample_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fallback labeling should mark all rows as review_label with zero confidence."""
    agent = _make_agent(temp_project)
    monkeypatch.setattr(agent, "_get_pipeline", lambda: None)

    labeled = agent.auto_label(sample_df)

    assert labeled["label"].eq("other_or_offtopic").all()
    assert labeled["confidence"].eq(0.0).all()
    assert labeled["label_source"].eq("fallback").all()


def test_flag_for_review_splits_correctly(temp_project: Path) -> None:
    """Rows should split into confident and review subsets by threshold."""
    agent = _make_agent(temp_project)
    df = pd.DataFrame(
        {
            "id": ["1", "2", "3", "4"],
            "text": ["a" * 60, "b" * 60, "c" * 60, "d" * 60],
            "source": ["s1", "s2", "s3", "s4"],
            "label": ["navigation", "safety", "equipment", "weather"],
            "confidence": [0.9, 0.5, 0.8, 0.3],
            "label_source": ["zero_shot"] * 4,
        }
    )

    df_confident, df_review = agent.flag_for_review(df)

    assert len(df_confident) == 2
    assert len(df_review) == 2
    assert df_confident["confidence"].tolist() == [0.9, 0.8]
    assert df_review["confidence"].tolist() == [0.5, 0.3]


def test_review_queue_saved(temp_project: Path) -> None:
    """Review queue CSV should be saved with corrected_label column."""
    agent = _make_agent(temp_project)
    df = pd.DataFrame(
        {
            "id": ["1", "2"],
            "text": ["a" * 60, "b" * 60],
            "source": ["s1", "s2"],
            "label": ["navigation", "weather"],
            "confidence": [0.2, 0.9],
            "label_source": ["zero_shot", "zero_shot"],
        }
    )

    agent.flag_for_review(df)
    review_queue = pd.read_csv(temp_project / "data" / "review_queue.csv")

    assert "corrected_label" in review_queue.columns


def test_check_quality_returns_metrics(temp_project: Path) -> None:
    """Quality metrics should include aggregate confidence and low-confidence counts."""
    agent = _make_agent(temp_project)
    df = pd.DataFrame(
        {
            "id": ["1", "2", "3"],
            "text": ["a" * 60, "b" * 60, "c" * 60],
            "source": ["s1", "s2", "s3"],
            "label": ["navigation", "other_or_offtopic", "safety"],
            "confidence": [0.9, 0.4, 0.8],
            "label_source": ["zero_shot", "fallback", "zero_shot"],
        }
    )

    metrics = agent.check_quality(df)

    assert {"total_labeled", "confidence_mean", "low_confidence_count"} <= metrics.keys()
    assert metrics["total_labeled"] == 3
    assert metrics["review_label_count"] == 1


def test_check_quality_includes_kappa(temp_project: Path) -> None:
    """Quality metrics should include Cohen's kappa when HITL corrections exist."""
    agent = _make_agent(temp_project)
    df = pd.DataFrame(
        {
            "id": ["1", "2", "3"],
            "text": ["a" * 60, "b" * 60, "c" * 60],
            "source": ["s1", "s2", "s3"],
            "label": ["navigation", "safety", "equipment"],
            "confidence": [0.9, 0.8, 0.7],
            "label_source": ["zero_shot", "zero_shot", "zero_shot"],
        }
    )
    review_queue = pd.DataFrame(
        {
            "id": ["1", "2"],
            "text": ["a" * 60, "b" * 60],
            "label": ["navigation", "safety"],
            "confidence": [0.9, 0.8],
            "source": ["s1", "s2"],
            "suggested_label": ["navigation", "safety"],
            "corrected_label": ["navigation", "equipment"],
        }
    )
    review_queue.to_csv(temp_project / "data" / "review_queue.csv", index=False, encoding="utf-8")

    metrics = agent.check_quality(df)

    assert "cohens_kappa" in metrics
    assert metrics["kappa_n_samples"] == 2


def test_kappa_not_calculated_without_hitl(temp_project: Path) -> None:
    """Kappa should remain unavailable when review_queue.csv is missing or empty."""
    agent = _make_agent(temp_project)
    df = pd.DataFrame(
        {
            "id": ["1", "2"],
            "text": ["a" * 60, "b" * 60],
            "source": ["s1", "s2"],
            "label": ["navigation", "safety"],
            "confidence": [0.9, 0.4],
            "label_source": ["zero_shot", "zero_shot"],
        }
    )

    metrics = agent.check_quality(df)

    assert metrics["cohens_kappa"] is None
    assert "not calculated" in metrics["kappa_interpretation"]


def test_generate_spec_saves_file(
    temp_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Generated annotation spec should be saved to reports/annotation_spec.md."""
    agent = _make_agent(temp_project)
    markdown = (
        "## Задача\nРазметка текстов.\n\n## Классы (с определениями)\n"
        "- navigation: навигация\n\n## Примеры (3+ на класс)\n- route\n\n## Граничные случаи\n- mixed"
    )
    monkeypatch.setattr(agent._llm_client, "generate", lambda prompt: markdown)

    result = agent.generate_spec()

    assert "## Задача" in result
    assert (temp_project / "reports" / "annotation_spec.md").exists()


def test_export_labelstudio_format(temp_project: Path) -> None:
    """Label Studio export should save a JSON list with id, data, and annotations."""
    agent = _make_agent(temp_project)
    df = pd.DataFrame(
        {
            "id": ["1"],
            "text": ["Annotated navigation text for label studio export."],
            "source": ["stackexchange_sailing"],
            "label": ["navigation"],
            "confidence": [0.88],
            "label_source": ["zero_shot"],
        }
    )

    agent.export_to_labelstudio(df)
    payload = json.loads(
        (temp_project / "reports" / "labelstudio_import.json").read_text(
            encoding="utf-8"
        )
    )

    assert isinstance(payload, list)
    assert {"id", "data", "annotations"} <= payload[0].keys()


def test_run_saves_annotated_parquet(
    temp_project: Path, sample_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run should save the fully annotated parquet file."""
    agent = _make_agent(temp_project)
    dataset_path = temp_project / "data" / "raw" / "dataset_clean.parquet"
    sample_df.to_parquet(dataset_path, index=False)
    monkeypatch.setattr(agent, "_get_pipeline", lambda: None)

    agent.run()

    assert (temp_project / "data" / "labeled" / "annotated.parquet").exists()


def test_context_memory_updated(
    temp_project: Path, sample_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run should update context memory to step 3.1."""
    agent = _make_agent(temp_project)
    dataset_path = temp_project / "data" / "raw" / "dataset_clean.parquet"
    sample_df.to_parquet(dataset_path, index=False)
    monkeypatch.setattr(agent, "_get_pipeline", lambda: None)

    agent.run()

    payload = json.loads(
        (temp_project / "reports" / "context_memory.json").read_text(encoding="utf-8")
    )
    assert payload["step"] == "3.1"
    assert payload["notes"] == "AnnotationAgent auto-labeled"


def test_summary_no_raw_data(
    temp_project: Path, sample_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Summary should remain compact and exclude raw text payloads."""
    agent = _make_agent(temp_project)
    dataset_path = temp_project / "data" / "raw" / "dataset_clean.parquet"
    sample_df.to_parquet(dataset_path, index=False)
    monkeypatch.setattr(agent, "_get_pipeline", lambda: None)

    agent.run()
    summary = agent.summary()
    encoded = json.dumps(summary, ensure_ascii=False)

    assert {"texts", "rows", "data"}.isdisjoint(summary.keys())
    assert "Route planning with waypoints" not in encoded
