"""Tests for ModelWrapper."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.model_wrapper import ModelWrapper


def _write_config(base_dir: Path, model_type: str = "sklearn") -> Path:
    """Create a minimal config file for model wrapper tests."""
    config = {
        "project": {"name": "smart-data-pipeline"},
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
        },
        "data": {
            "raw_path": "data/raw",
            "labeled_path": "data/labeled",
            "review_queue_path": "data/review_queue.csv",
        },
        "model": {
            "type": model_type,
            "base_model": "logistic_regression",
            "exclude_offtopic": True,
            "test_size": 0.2,
            "random_state": 42,
            "tfidf": {
                "max_features": 10000,
                "ngram_range": [1, 2],
                "sublinear_tf": True,
            },
            "logreg": {
                "max_iter": 1000,
                "class_weight": "balanced",
                "C": 1.0,
            },
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
    """Create an isolated project tree for model wrapper tests."""
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "labeled").mkdir(parents=True)
    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / "reports").mkdir(parents=True)
    _write_config(tmp_path)
    return tmp_path


@pytest.fixture
def labeled_df() -> pd.DataFrame:
    """Return a synthetic multi-class dataset with balanced samples."""
    rows: list[dict[str, str]] = []
    class_keywords = {
        "navigation": "chart route compass heading waypoint harbor navigation",
        "safety": "lifejacket rescue man overboard distress flare safety",
        "equipment": "winch sail rigging mast anchor equipment maintenance",
        "weather": "forecast wind storm pressure tide weather clouds",
        "licensing": "license exam certification permit regulation training",
        "other_or_offtopic": "happy feelings social tweet emotion random chatter",
    }
    for label, keywords in class_keywords.items():
        for index in range(10):
            rows.append(
                {
                    "id": f"{label}-{index}",
                    "text": f"{keywords} sample text {index} for {label}",
                    "label": label,
                    "source": "synthetic",
                }
            )
    return pd.DataFrame(rows)


def _make_wrapper(temp_project: Path, model_type: str = "sklearn") -> ModelWrapper:
    """Build a ModelWrapper pointed at the temporary config."""
    _write_config(temp_project, model_type=model_type)
    return ModelWrapper(config_path=str(temp_project / "config.yaml"))


def test_wrapper_initializes(temp_project: Path) -> None:
    """Wrapper should initialize config and stay lazy before training."""
    wrapper = _make_wrapper(temp_project)

    assert wrapper.model_type == "sklearn"
    assert wrapper._model is None
    assert wrapper.summary()["trained"] is False


def test_fit_returns_metrics(temp_project: Path, labeled_df: pd.DataFrame) -> None:
    """fit() should return the core evaluation metrics."""
    wrapper = _make_wrapper(temp_project)

    metrics = wrapper.fit(labeled_df)

    assert {"accuracy", "f1_macro", "f1_weighted", "f1_per_class"} <= metrics.keys()


def test_predict_returns_labels(temp_project: Path, labeled_df: pd.DataFrame) -> None:
    """predict() should return string labels after fitting."""
    wrapper = _make_wrapper(temp_project)
    wrapper.fit(labeled_df)

    predicted = wrapper.predict(["chart route compass and harbor navigation"])

    assert len(predicted) == 1
    assert isinstance(predicted[0], str)


def test_predict_proba_keys(temp_project: Path, labeled_df: pd.DataFrame) -> None:
    """predict_proba() should return class labels and probability matrix."""
    wrapper = _make_wrapper(temp_project)
    wrapper.fit(labeled_df)

    result = wrapper.predict_proba(["storm pressure weather forecast"])

    assert {"labels", "probas"} <= result.keys()
    assert len(result["probas"]) == 1


def test_evaluate_returns_all_keys(
    temp_project: Path, labeled_df: pd.DataFrame
) -> None:
    """evaluate() should expose the full evaluation payload."""
    wrapper = _make_wrapper(temp_project)
    wrapper.fit(labeled_df)
    X_test = [
        "harbor route chart navigation",
        "distress flare rescue safety",
        "license exam permit licensing",
    ]
    classes = list(wrapper._label_encoder.classes_)
    y_test = wrapper._label_encoder.transform(
        [
            "navigation" if "navigation" in classes else classes[0],
            "safety" if "safety" in classes else classes[0],
            "licensing" if "licensing" in classes else classes[0],
        ]
    )

    metrics = wrapper.evaluate(X_test, y_test)

    assert {
        "accuracy",
        "f1_macro",
        "f1_weighted",
        "f1_per_class",
        "classification_report",
        "n_test",
    } <= metrics.keys()


def test_save_and_load(temp_project: Path, labeled_df: pd.DataFrame) -> None:
    """A saved model should be loadable into a fresh wrapper instance."""
    wrapper = _make_wrapper(temp_project)
    wrapper.fit(labeled_df)
    wrapper.save()

    reloaded = _make_wrapper(temp_project)
    reloaded.load()
    predicted = reloaded.predict(["anchor rigging mast equipment check"])

    assert len(predicted) == 1
    assert isinstance(predicted[0], str)


def test_distilbert_stub_raises(temp_project: Path, labeled_df: pd.DataFrame) -> None:
    """DistilBERT mode should raise the planned stub error."""
    wrapper = _make_wrapper(temp_project, model_type="distilbert")

    with pytest.raises(NotImplementedError):
        wrapper.fit(labeled_df)


def test_explain_returns_list(temp_project: Path, labeled_df: pd.DataFrame) -> None:
    """explain() should return feature contributions for each text."""
    wrapper = _make_wrapper(temp_project)
    wrapper.fit(labeled_df)

    explanations = wrapper.explain(["sailing chart route compass navigation"], n_features=5)

    assert len(explanations) == 1
    assert "top_features" in explanations[0]


def test_summary_trained(temp_project: Path, labeled_df: pd.DataFrame) -> None:
    """summary() should mark the wrapper as trained after fit()."""
    wrapper = _make_wrapper(temp_project)
    wrapper.fit(labeled_df)

    summary = wrapper.summary()

    assert summary["trained"] is True
    assert summary["n_classes"] >= 5


def test_model_saved_to_disk(temp_project: Path, labeled_df: pd.DataFrame) -> None:
    """fit() should persist the trained classifier artifact."""
    wrapper = _make_wrapper(temp_project)
    wrapper.fit(labeled_df)

    assert (temp_project / "models" / "classifier.pkl").exists()
