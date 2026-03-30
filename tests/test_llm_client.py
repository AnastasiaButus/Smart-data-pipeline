"""Tests for GeminiLLMClient and ContextMemory."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.context_memory import ContextMemory
from core.llm_client import GeminiLLMClient


def _write_config(base_dir: Path, max_prompt_chars: int = 2000) -> Path:
    """Create a minimal config file for llm client tests."""
    config = {
        "project": {
            "name": "smart-data-pipeline",
            "version": "0.1.0",
            "description": "Test config",
        },
        "domain": {
            "topic": "sailing and yacht navigation",
            "normalized_topic": "",
            "classes": [
                "navigation",
                "safety",
                "equipment",
                "weather",
                "licensing",
            ],
            "keywords_by_class": {},
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
            "max_prompt_chars": max_prompt_chars,
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
    """Create an isolated project directory for llm tests."""
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "reports").mkdir(parents=True)
    _write_config(tmp_path)
    return tmp_path


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Build a small deterministic dataframe for summary tests."""
    return pd.DataFrame(
        {
            "id": ["1", "2", "3"],
            "text": [
                "Celestial navigation uses charts, bearings, and weather routing.",
                "Life jackets, MOB drills, and EPIRB checks matter for offshore safety.",
                "Storm jib setup and barometer tracking support passage planning &amp; crew prep.",
            ],
            "label": ["unlabeled", "unlabeled", "unlabeled"],
            "source": [
                "stackexchange_sailing",
                "rss_example",
                "huggingface_example",
            ],
            "collected_at": [
                "2026-03-30T10:00:00+00:00",
                "2026-03-30T10:01:00+00:00",
                "2026-03-30T10:02:00+00:00",
            ],
        }
    )


def _valid_spec() -> dict:
    """Return a valid mocked domain specification."""
    return {
        "normalized_topic": "sailing domain taxonomy for navigation and onboard operations",
        "recommended_classes": [
            "navigation",
            "safety",
            "equipment",
            "weather",
            "licensing",
        ],
        "keywords_by_class": {
            "navigation": ["waypoint", "chart", "route"],
            "safety": ["mob", "lifejacket", "distress"],
            "equipment": ["rigging", "anchor", "engine"],
            "weather": ["forecast", "wind", "storm"],
            "licensing": ["certificate", "radio", "regulation"],
        },
        "collection_queries": {
            "navigation": ["sailing route planning", "chartplotter waypoint setup"],
            "safety": ["man overboard drill sailing", "offshore safety gear"],
            "equipment": ["sailboat rigging maintenance", "marine anchor setup"],
            "weather": ["marine forecast sailing", "barometer storm sailing"],
            "licensing": ["boating license sailing", "vhf radio certificate"],
        },
        "source_fit": {
            "huggingface_example": {
                "fit": "low",
                "notes": "Noisy sentiment-like data with partial domain mismatch.",
            },
            "stackexchange_sailing": {
                "fit": "high",
                "notes": "Operational sailing questions match the domain well.",
            },
            "rss_example": {
                "fit": "medium",
                "notes": "Editorial content is useful but can be headline-heavy.",
            },
        },
        "review_label": "other_or_offtopic",
        "annotation_guidelines": [
            "Prefer the concrete sailing topic over tone or sentiment.",
            "Send off-topic social chatter to other_or_offtopic.",
        ],
        "risks": [
            "HuggingFace rows remain partially off-topic.",
            "Weather and navigation may overlap on some passages.",
        ],
        "llm_notes": "Mocked Gemini output for tests.",
    }


def test_client_initializes(temp_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Client should initialize without creating a network client eagerly."""
    monkeypatch.setenv("GEMINI_API_KEY", "")
    client = GeminiLLMClient(config_path=str(temp_project / "config.yaml"))
    assert client._config_valid is True
    assert client._client is None
    assert client._max_prompt_chars == 2000


def test_is_available_without_key(
    temp_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Availability should be false when the API key is absent."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    client = GeminiLLMClient(config_path=str(temp_project / "config.yaml"))
    assert client.is_available() is False


def test_build_dataset_summary_no_raw_texts(
    temp_project: Path, sample_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Summary must not expose raw texts or forbidden data keys."""
    monkeypatch.setenv("GEMINI_API_KEY", "")
    client = GeminiLLMClient(config_path=str(temp_project / "config.yaml"))
    sample_df.loc[0, "text"] = (
        "This is a raw text sentence that should never appear verbatim in the summary."
    )

    summary = client.build_dataset_summary(sample_df)
    encoded = json.dumps(summary, ensure_ascii=False)

    forbidden_keys = {"examples", "rows", "texts", "data"}
    assert forbidden_keys.isdisjoint(summary.keys())
    assert "This is a raw text sentence that should never appear verbatim in the summary." not in encoded


def test_build_dataset_summary_expected_keys(
    temp_project: Path, sample_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Summary should include all expected compact aggregate keys."""
    monkeypatch.setenv("GEMINI_API_KEY", "")
    client = GeminiLLMClient(config_path=str(temp_project / "config.yaml"))
    summary = client.build_dataset_summary(sample_df)

    expected_keys = {
        "total_rows",
        "columns",
        "source_distribution",
        "avg_text_len",
        "min_text_len",
        "max_text_len",
        "pct_short_texts",
        "pct_long_texts",
        "html_entity_count",
        "top_keywords_global",
        "top_keywords_by_source",
        "current_topic",
        "current_classes",
    }
    assert expected_keys.issubset(summary.keys())
    assert summary["total_rows"] == 3


def test_heuristic_fallback_returns_valid_schema(
    temp_project: Path, sample_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Heuristic fallback should return the required schema and source-fit logic."""
    monkeypatch.setenv("GEMINI_API_KEY", "")
    client = GeminiLLMClient(config_path=str(temp_project / "config.yaml"))
    summary = client.build_dataset_summary(sample_df)

    spec = client.heuristic_fallback(
        topic="sailing and yacht navigation",
        current_classes=["navigation", "safety", "equipment", "weather", "licensing"],
        dataset_summary=summary,
    )

    expected_keys = {
        "normalized_topic",
        "recommended_classes",
        "keywords_by_class",
        "collection_queries",
        "source_fit",
        "review_label",
        "annotation_guidelines",
        "risks",
        "llm_notes",
    }
    assert expected_keys.issubset(spec.keys())
    assert spec["review_label"] == "other_or_offtopic"
    assert spec["source_fit"]["huggingface_example"]["fit"] in {"low", "medium"}
    assert spec["source_fit"]["stackexchange_sailing"]["fit"] in {"medium", "high"}
    assert spec["source_fit"]["rss_example"]["fit"] in {"medium", "high"}


def test_generate_domain_spec_with_mocked_gemini(
    temp_project: Path, sample_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Domain spec generation should use mocked Gemini output and validate it."""
    client = GeminiLLMClient(config_path=str(temp_project / "config.yaml"))
    summary = client.build_dataset_summary(sample_df)
    monkeypatch.setattr(client, "is_available", lambda: True)
    monkeypatch.setattr(client, "_generate_json_response", lambda prompt: _valid_spec())

    spec = client.generate_domain_spec(
        topic="sailing and yacht navigation",
        dataset_summary=summary,
        current_classes=["navigation", "safety", "equipment", "weather", "licensing"],
    )

    assert spec["normalized_topic"].startswith("sailing domain taxonomy")
    assert spec["review_label"] == "other_or_offtopic"
    assert len(spec["recommended_classes"]) == 5


def test_update_config_domain(
    temp_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Config update should preserve the original topic and add new fields."""
    monkeypatch.setenv("GEMINI_API_KEY", "")
    config_path = temp_project / "config.yaml"
    client = GeminiLLMClient(config_path=str(config_path))

    client.update_config_domain(_valid_spec())

    updated = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert updated["domain"]["topic"] == "sailing and yacht navigation"
    assert updated["domain"]["normalized_topic"] == _valid_spec()["normalized_topic"]
    assert updated["domain"]["review_label"] == "other_or_offtopic"
    assert updated["domain"]["keywords_by_class"]["navigation"] == [
        "waypoint",
        "chart",
        "route",
    ]
    assert updated["llm"]["max_prompt_chars"] == 2000


def test_run_from_dataset_saves_reports(
    temp_project: Path, sample_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end run should save reports, update config, and update context memory."""
    monkeypatch.setenv("GEMINI_API_KEY", "")
    dataset_path = temp_project / "data" / "raw" / "dataset.parquet"
    sample_df.to_parquet(dataset_path, index=False)

    client = GeminiLLMClient(config_path=str(temp_project / "config.yaml"))
    monkeypatch.setattr(client, "generate_domain_spec", lambda *args, **kwargs: _valid_spec())

    spec = client.run_from_dataset(dataset_path=str(dataset_path))

    assert spec["review_label"] == "other_or_offtopic"
    assert (temp_project / "reports" / "domain_spec.json").exists()
    assert (temp_project / "reports" / "domain_reformulation.md").exists()
    memory = json.loads((temp_project / "reports" / "context_memory.json").read_text(encoding="utf-8"))
    assert memory["step"] == "1.3"
    assert memory["metrics"]["recommended_classes_count"] == 5


def test_context_memory_step_13(temp_project: Path) -> None:
    """Step 1.3 memory updates should keep the current step in top-level fields."""
    memory = ContextMemory(path=str(temp_project / "reports" / "context_memory.json"))
    memory.update(
        step="1.3",
        status="done",
        metrics={
            "total_rows": 761,
            "source_distribution": {"huggingface_example": 10},
            "recommended_classes_count": 5,
            "review_label": "other_or_offtopic",
        },
        notes="Domain spec generated with Gemini",
    )

    payload = json.loads((temp_project / "reports" / "context_memory.json").read_text(encoding="utf-8"))
    assert payload["step"] == "1.3"
    assert payload["status"] == "done"
    assert payload["metrics"]["review_label"] == "other_or_offtopic"


def test_context_memory_update_and_read(temp_project: Path) -> None:
    """ContextMemory should save and expose historical step records."""
    path = temp_project / "reports" / "context_memory.json"
    memory = ContextMemory(path=str(path))
    memory.update("1.2", "done", {"total_rows": 761, "sources": {"rss": 60}}, "collection done")
    memory.update("1.3", "done", {"total_rows": 761, "recommended_classes_count": 5}, "spec done")

    assert memory.get("1.2")["notes"] == "collection done"
    assert memory.get("1.3")["metrics"]["recommended_classes_count"] == 5
    assert "history" in memory.get()


def test_context_memory_summary_length(temp_project: Path) -> None:
    """Context summary should remain compact for prompt use."""
    path = temp_project / "reports" / "context_memory.json"
    memory = ContextMemory(path=str(path))
    for index in range(5):
        memory.update(
            step=f"1.{index}",
            status="done",
            metrics={
                "total_rows": 100 + index,
                "source_distribution": {"rss_example": 10 + index},
                "recommended_classes_count": 5,
                "review_label": "other_or_offtopic",
            },
            notes="x" * 80,
        )

    summary = memory.get_summary_for_llm(max_chars=120)
    assert len(summary) <= 120
    assert "other_or_offtopic" in summary or summary.endswith("...")


def test_prompt_truncation_or_prompt_limit_behavior(
    tmp_path: Path, sample_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Built prompt should respect ``max_prompt_chars`` even for large summaries."""
    monkeypatch.setenv("GEMINI_API_KEY", "")
    config_path = _write_config(tmp_path, max_prompt_chars=900)
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)
    client = GeminiLLMClient(config_path=str(config_path))

    large_df = pd.concat([sample_df] * 20, ignore_index=True)
    summary = client.build_dataset_summary(large_df)
    prompt = client._build_prompt(
        topic="sailing and yacht navigation",
        dataset_summary=summary,
        current_classes=["navigation", "safety", "equipment", "weather", "licensing"],
    )

    assert len(prompt) <= 900


def test_generate_eda_hypotheses_returns_list(
    temp_project: Path, sample_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EDA hypotheses should return a five-item string list from mocked Gemini output."""
    client = GeminiLLMClient(config_path=str(temp_project / "config.yaml"))
    summary = client.build_dataset_summary(sample_df)
    monkeypatch.setattr(client, "is_available", lambda: True)
    monkeypatch.setattr(
        client,
        "generate",
        lambda prompt: json.dumps(
            [
                "Hypothesis 1",
                "Hypothesis 2",
                "Hypothesis 3",
                "Hypothesis 4",
                "Hypothesis 5",
            ]
        ),
    )

    hypotheses = client.generate_eda_hypotheses(summary)

    assert isinstance(hypotheses, list)
    assert len(hypotheses) == 5


def test_generate_eda_hypotheses_fallback(
    temp_project: Path, sample_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EDA hypotheses should fall back to heuristics when Gemini fails."""
    client = GeminiLLMClient(config_path=str(temp_project / "config.yaml"))
    summary = client.build_dataset_summary(sample_df)
    monkeypatch.setattr(client, "is_available", lambda: True)

    def _raise(_: str) -> str:
        raise Exception("gemini failed")

    monkeypatch.setattr(client, "generate", _raise)

    hypotheses = client.generate_eda_hypotheses(summary)

    assert isinstance(hypotheses, list)
    assert len(hypotheses) == 5


def test_generate_stopwords_returns_set(
    temp_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LLM stopwords should merge parsed Gemini words with the base set."""
    client = GeminiLLMClient(config_path=str(temp_project / "config.yaml"))
    base_stopwords = {"href", "html", "content"}
    monkeypatch.setattr(
        client,
        "generate_json",
        lambda prompt: {"stopwords": ["sail", "boat"]},
    )

    stopwords = client.generate_stopwords("sailing and yacht navigation", base_stopwords)

    assert isinstance(stopwords, set)
    assert "sail" in stopwords
    assert "boat" in stopwords
    assert base_stopwords.issubset(stopwords)


def test_generate_stopwords_fallback(
    temp_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stopwords generation should fall back to the base set when Gemini fails."""
    client = GeminiLLMClient(config_path=str(temp_project / "config.yaml"))
    base_stopwords = {"href", "html", "content"}

    def _raise(_: str) -> dict[str, list[str]]:
        raise Exception("gemini stopwords failed")

    monkeypatch.setattr(client, "generate_json", _raise)

    stopwords = client.generate_stopwords("sailing and yacht navigation", base_stopwords)

    assert stopwords == base_stopwords
