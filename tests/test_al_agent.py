"""Tests for ActiveLearningAgent."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from agents.al_agent import ActiveLearningAgent


def _write_config(base_dir: Path) -> Path:
    """Create a minimal config file for Active Learning tests."""
    config = {
        "project": {"name": "smart-data-pipeline"},
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
        "active_learning": {
            "strategy": "entropy",
            "initial_size": 10,
            "batch_size": 5,
            "n_iterations": 2,
            "compare_strategies": True,
        },
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
    (tmp_path / "data" / "labeled").mkdir(parents=True)
    (tmp_path / "reports").mkdir(parents=True)
    _write_config(tmp_path)
    return tmp_path


@pytest.fixture
def labeled_df() -> pd.DataFrame:
    """Return a compact multi-class labeled dataframe."""
    classes = ["navigation", "safety", "equipment", "weather", "licensing"]
    rows: list[dict[str, str]] = []
    for class_name in classes:
        for index in range(12):
            rows.append(
                {
                    "id": f"{class_name}-{index}",
                    "text": f"{class_name} sailing sample text number {index} with domain vocabulary",
                    "label": class_name,
                    "source": "annotated",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def pool_df() -> pd.DataFrame:
    """Return a review-pool dataframe with predicted labels."""
    classes = ["navigation", "safety", "equipment", "weather", "licensing"]
    rows: list[dict[str, str | float]] = []
    for index in range(100):
        class_name = classes[index % len(classes)]
        rows.append(
            {
                "id": f"pool-{index}",
                "text": f"pool text {index} about {class_name} and offshore sailing review",
                "label": class_name,
                "confidence": 0.4 + ((index % 5) * 0.1),
                "source": "review_queue",
                "suggested_label": class_name,
                "corrected_label": "",
            }
        )
    return pd.DataFrame(rows)


def _make_agent(temp_project: Path) -> ActiveLearningAgent:
    """Build an ActiveLearningAgent pointed at the temp project config."""
    return ActiveLearningAgent(config_path=str(temp_project / "config.yaml"))


def test_agent_initializes(temp_project: Path) -> None:
    """Agent should initialize config and lazy model state."""
    agent = _make_agent(temp_project)

    assert agent._model is None
    assert agent._strategy == "entropy"
    assert agent._initial_size == 10
    assert agent._batch_size == 5


def test_fit_trains_model(temp_project: Path, labeled_df: pd.DataFrame) -> None:
    """fit() should train and store the underlying model."""
    agent = _make_agent(temp_project)

    agent.fit(labeled_df.iloc[:50])

    assert agent._model is not None


def test_predict_proba_shape(temp_project: Path, labeled_df: pd.DataFrame) -> None:
    """predict_proba() should return (n_samples, n_classes) after fit."""
    agent = _make_agent(temp_project)
    agent.fit(labeled_df.iloc[:50])

    proba = agent.predict_proba(["sample one", "sample two", "sample three"])

    assert proba.shape == (3, len(agent._classes_))


def test_query_entropy_returns_n_rows(
    temp_project: Path, labeled_df: pd.DataFrame, pool_df: pd.DataFrame
) -> None:
    """Entropy query should return exactly n rows."""
    agent = _make_agent(temp_project)
    agent.fit(labeled_df)

    queried = agent.query(pool_df, "entropy", n=10)

    assert len(queried) == 10


def test_query_margin_returns_n_rows(
    temp_project: Path, labeled_df: pd.DataFrame, pool_df: pd.DataFrame
) -> None:
    """Margin query should return exactly n rows."""
    agent = _make_agent(temp_project)
    agent.fit(labeled_df)

    queried = agent.query(pool_df, "margin", n=10)

    assert len(queried) == 10


def test_query_random_returns_n_rows(temp_project: Path, pool_df: pd.DataFrame) -> None:
    """Random query should return exactly n rows without needing a fitted model."""
    agent = _make_agent(temp_project)

    queried = agent.query(pool_df, "random", n=10)

    assert len(queried) == 10


def test_evaluate_returns_metrics(temp_project: Path, labeled_df: pd.DataFrame) -> None:
    """evaluate() should return the expected metric keys."""
    agent = _make_agent(temp_project)

    metrics = agent.evaluate(labeled_df)

    assert {"accuracy", "f1_macro", "f1_weighted", "f1_per_class"} <= metrics.keys()


def test_run_cycle_returns_history(
    temp_project: Path, labeled_df: pd.DataFrame, pool_df: pd.DataFrame
) -> None:
    """run_cycle() should return one history record per iteration."""
    agent = _make_agent(temp_project)

    history = agent.run_cycle(labeled_df, pool_df, n_iterations=2, batch_size=5)

    assert len(history) == 2
    assert {"iteration", "n_labeled", "accuracy"} <= history[0].keys()


def test_history_grows_with_iterations(
    temp_project: Path, labeled_df: pd.DataFrame, pool_df: pd.DataFrame
) -> None:
    """n_labeled should grow across AL iterations."""
    agent = _make_agent(temp_project)

    history = agent.run_cycle(labeled_df, pool_df, n_iterations=2, batch_size=5)

    assert history[1]["n_labeled"] > history[0]["n_labeled"]


def test_report_saves_files(temp_project: Path) -> None:
    """report() should save the interactive learning curve HTML."""
    agent = _make_agent(temp_project)
    history = [
        {"iteration": 1, "n_labeled": 10, "accuracy": 0.5, "f1_macro": 0.45, "strategy": "entropy"},
        {"iteration": 2, "n_labeled": 15, "accuracy": 0.6, "f1_macro": 0.55, "strategy": "entropy"},
    ]

    agent.report(history)

    assert (temp_project / "reports" / "learning_curve.html").exists()


def test_summary_no_raw_data(
    temp_project: Path, labeled_df: pd.DataFrame, pool_df: pd.DataFrame
) -> None:
    """summary() should not expose raw text payloads."""
    agent = _make_agent(temp_project)
    agent.run_cycle(labeled_df, pool_df, n_iterations=2, batch_size=5)
    agent._last_summary = {
        "strategy": "entropy",
        "n_iterations": 2,
        "final_accuracy": 0.6,
        "final_f1": 0.55,
        "best_strategy": "margin",
    }

    summary = agent.summary()
    encoded = json.dumps(summary, ensure_ascii=False)

    assert {"texts", "rows", "data"}.isdisjoint(summary.keys())
    assert "sailing sample text" not in encoded


def test_context_memory_updated(
    temp_project: Path, labeled_df: pd.DataFrame, pool_df: pd.DataFrame
) -> None:
    """run() should update context memory for step 4.1."""
    agent = _make_agent(temp_project)
    labeled_df.to_parquet(temp_project / "data" / "labeled" / "annotated.parquet", index=False)
    pool_df.to_csv(temp_project / "data" / "review_queue.csv", index=False)

    result = agent.run()
    payload = json.loads((temp_project / "reports" / "context_memory.json").read_text(encoding="utf-8"))

    assert result["strategy"] == "entropy"
    assert payload["step"] == "4.1"
    assert payload["notes"] == "AL cycle completed"
