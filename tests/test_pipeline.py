"""Tests for the Prefect pipeline orchestration layer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pipeline.run_pipeline as run_pipeline_module
from pipeline.run_pipeline import annotate, clean, collect, data_pipeline, human_review, train
import core.model_wrapper as model_wrapper_module


@pytest.fixture
def temp_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create an isolated workspace for pipeline orchestration tests."""
    (tmp_path / "data").mkdir(parents=True)
    (tmp_path / "reports").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _sample_confident_df() -> pd.DataFrame:
    """Return a compact dataframe compatible with downstream training."""
    return pd.DataFrame(
        {
            "id": ["1", "2"],
            "text": ["navigation text", "safety text"],
            "label": ["navigation", "safety"],
            "confidence": [0.91, 0.88],
            "source": ["stackexchange_sailing", "stackexchange_sailing"],
            "label_source": ["zero_shot", "zero_shot"],
        }
    )


def test_pipeline_imports_cleanly() -> None:
    """Pipeline module objects should import without errors."""
    assert data_pipeline is not None
    assert collect is not None
    assert clean is not None
    assert annotate is not None
    assert train is not None


def test_pipeline_tasks_are_decorated() -> None:
    """Core pipeline steps should be Prefect task objects."""
    for task_obj in (collect, clean, annotate, train):
        assert hasattr(task_obj, "fn")
        assert callable(task_obj.fn)


def test_human_review_with_empty_queue(temp_project: Path) -> None:
    """Without a review queue file the confident dataframe should pass through unchanged."""
    df_confident = _sample_confident_df()

    result = human_review.fn(df_confident, None)

    pd.testing.assert_frame_equal(result.reset_index(drop=True), df_confident)


def test_human_review_with_corrections(temp_project: Path) -> None:
    """Corrected review rows should be merged back into the confident dataframe."""
    df_confident = _sample_confident_df()
    review_path = temp_project / "data" / "review_queue.csv"
    review_df = pd.DataFrame(
        {
            "id": ["3"],
            "text": ["weather forecast text"],
            "label": ["weather"],
            "confidence": [0.41],
            "source": ["rss_www.yachtingworld.com"],
            "suggested_label": ["weather"],
            "corrected_label": ["weather"],
        }
    )
    review_df.to_csv(review_path, index=False)

    result = human_review.fn(df_confident, None)

    assert len(result) == 3
    assert "hitl" in result["label_source"].tolist()
    assert (temp_project / "reports" / "context_memory.json").exists()


def test_pipeline_skip_flags(
    temp_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flow should honor skip flags when downstream tasks are mocked."""
    sample_df = _sample_confident_df()
    calls = {"collect": 0, "clean": 0, "annotate": 0, "train": 0}

    monkeypatch.setattr(
        run_pipeline_module,
        "collect",
        lambda: calls.__setitem__("collect", calls["collect"] + 1) or sample_df,
    )
    monkeypatch.setattr(
        run_pipeline_module,
        "clean",
        lambda df: calls.__setitem__("clean", calls["clean"] + 1) or df,
    )
    monkeypatch.setattr(
        run_pipeline_module,
        "annotate",
        lambda df: calls.__setitem__("annotate", calls["annotate"] + 1) or df,
    )

    def _fail_human_review(*args, **kwargs):
        raise AssertionError("human_review should not be called when skip_hitl=True")

    def _fail_active_learn(*args, **kwargs):
        raise AssertionError("active_learn should not be called when skip_al=True")

    monkeypatch.setattr(run_pipeline_module, "human_review", _fail_human_review)
    monkeypatch.setattr(run_pipeline_module, "active_learn", _fail_active_learn)
    monkeypatch.setattr(
        run_pipeline_module,
        "train",
        lambda df: calls.__setitem__("train", calls["train"] + 1)
        or {"accuracy": 0.5, "f1_macro": 0.43},
    )

    result = data_pipeline.fn(skip_hitl=True, skip_al=True)

    assert result["accuracy"] == 0.5
    assert calls == {"collect": 1, "clean": 1, "annotate": 1, "train": 1}
    payload = json.loads((temp_project / "reports" / "context_memory.json").read_text(encoding="utf-8"))
    assert payload["step"] == "6.1"


def test_pipeline_topic_override_updates_config(
    temp_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Topic override should persist both the selected topic and the fresh artifact topic."""
    sample_df = _sample_confident_df()
    monkeypatch.setattr(run_pipeline_module, "CONFIG_PATH", temp_project / "config.yaml")
    (temp_project / "config.yaml").write_text(
        yaml.safe_dump({"domain": {"topic": "sailing and yacht navigation"}}, sort_keys=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(run_pipeline_module, "collect", lambda: sample_df)
    monkeypatch.setattr(run_pipeline_module, "clean", lambda df: df)
    monkeypatch.setattr(run_pipeline_module, "annotate", lambda df: df)
    monkeypatch.setattr(run_pipeline_module, "human_review", lambda df, _: df)
    monkeypatch.setattr(run_pipeline_module, "active_learn", lambda df: {"skipped": True})
    monkeypatch.setattr(
        run_pipeline_module,
        "train",
        lambda df: {"accuracy": 0.5, "f1_macro": 0.43},
    )

    result = data_pipeline.fn(skip_hitl=False, skip_al=True, topic="minecraft")

    assert result["accuracy"] == 0.5
    cfg = yaml.safe_load((temp_project / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["domain"]["topic"] == "minecraft"
    assert cfg["domain"]["normalized_topic"] == "minecraft"


def test_train_returns_skipped_metrics_for_insufficient_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Train task should return a warning payload when the reviewed dataset is not trainable yet."""

    def _raise_not_ready(self, df):
        raise ValueError("No labeled rows available for model training")

    monkeypatch.setattr(model_wrapper_module.ModelWrapper, "fit", _raise_not_ready)

    result = train.fn(_sample_confident_df())

    assert result["skipped"] is True
    assert "No labeled rows available" in result["reason"]
    assert result["accuracy"] == 0.0
