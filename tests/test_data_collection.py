"""Tests for DataCollectionAgent — schema, deduplication, graceful degradation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Ensure project root is on the path when running from repo root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.data_collection_agent import DataCollectionAgent

CONFIG_PATH = str(Path(__file__).parent.parent / "config.yaml")
REQUIRED_COLUMNS = ["id", "text", "label", "source", "collected_at"]


@pytest.fixture
def agent() -> DataCollectionAgent:
    """Create a DataCollectionAgent pointed at the real config."""
    return DataCollectionAgent(config_path=CONFIG_PATH)


# ------------------------------------------------------------------ #
#  1. Initialisation                                                   #
# ------------------------------------------------------------------ #

def test_agent_initializes(agent: DataCollectionAgent) -> None:
    """Agent must initialise without raising and expose a config dict."""
    assert agent._cfg is not None
    assert "sources" in agent._cfg


# ------------------------------------------------------------------ #
#  2. Output schema                                                    #
# ------------------------------------------------------------------ #

def test_output_schema(agent: DataCollectionAgent) -> None:
    """merge() output must have exactly the required columns."""
    df_in = pd.DataFrame({
        "text": ["This is a valid sailing text for the test"],
        "label": ["unlabeled"],
        "source": ["test"],
        "collected_at": ["2024-01-01T00:00:00+00:00"],
    })
    result = agent.merge([df_in])
    assert list(result.columns) == REQUIRED_COLUMNS


# ------------------------------------------------------------------ #
#  3. No duplicates after merge                                        #
# ------------------------------------------------------------------ #

def test_no_duplicates(agent: DataCollectionAgent) -> None:
    """merge() must remove duplicate texts (case/whitespace insensitive)."""
    row = {"text": "  Anchoring in a crowded bay requires attention  ",
           "label": "unlabeled", "source": "test",
           "collected_at": "2024-01-01T00:00:00+00:00"}
    df = pd.DataFrame([row, row, row])
    result = agent.merge([df])
    assert len(result) == 1


# ------------------------------------------------------------------ #
#  4. Short text filter                                                #
# ------------------------------------------------------------------ #

def test_merge_filters_short_texts(agent: DataCollectionAgent) -> None:
    """merge() must drop rows where text length < 20 characters."""
    rows = [
        {"text": "Too short", "label": "unlabeled", "source": "test", "collected_at": "2024-01-01"},
        {"text": "This text is long enough to pass the filter check", "label": "unlabeled",
         "source": "test", "collected_at": "2024-01-01"},
    ]
    result = agent.merge([pd.DataFrame(rows)])
    assert all(result["text"].str.len() >= 20)
    assert len(result) == 1


# ------------------------------------------------------------------ #
#  5. HuggingFace fetch                                                #
# ------------------------------------------------------------------ #

def test_huggingface_fetch(agent: DataCollectionAgent) -> None:
    """fetch_huggingface() must return a non-empty DataFrame with correct schema."""
    result = agent.fetch_huggingface()
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0
    for col in ["text", "label", "source", "collected_at"]:
        assert col in result.columns, f"Missing column: {col}"
    assert result["label"].eq("unlabeled").all()
    assert result["source"].str.startswith("huggingface_").all()


# ------------------------------------------------------------------ #
#  6. RSS graceful degradation                                         #
# ------------------------------------------------------------------ #

def test_rss_fetch_graceful(agent: DataCollectionAgent) -> None:
    """fetch_rss() must not raise even if all feeds are unreachable."""
    import feedparser

    def _mock_parse(url: str, *args, **kwargs) -> MagicMock:
        mock = MagicMock()
        mock.bozo = True
        mock.entries = []
        return mock

    with patch.object(feedparser, "parse", side_effect=_mock_parse):
        result = agent.fetch_rss()

    assert isinstance(result, pd.DataFrame)
    # May be empty — that's fine; what matters is no exception raised


# ------------------------------------------------------------------ #
#  7. Forum scrape does not crash on network failure                   #
# ------------------------------------------------------------------ #

def test_forum_scrape_checks_robots(agent: DataCollectionAgent) -> None:
    """scrape_forum() must return empty DataFrame when site is unreachable."""
    with patch.object(agent, "_is_crawl_allowed", return_value=False):
        result = agent.scrape_forum()
    assert isinstance(result, pd.DataFrame)
    assert result.empty


# ------------------------------------------------------------------ #
#  8. Synthetic fallback                                               #
# ------------------------------------------------------------------ #

def test_synthetic_fallback(agent: DataCollectionAgent) -> None:
    """_generate_synthetic(n) must return exactly n rows with correct columns."""
    result = agent._generate_synthetic(n=10)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 10
    assert list(result.columns) == REQUIRED_COLUMNS
    assert result["source"].eq("synthetic").all()
    assert result["label"].eq("unlabeled").all()


# ------------------------------------------------------------------ #
#  9. Context memory saved after run                                   #
# ------------------------------------------------------------------ #

def test_context_memory_saved(agent: DataCollectionAgent, tmp_path: Path) -> None:
    """run() must write context_memory.json with a 'metrics' key."""
    # Patch all network sources to avoid real HTTP calls in CI
    empty = pd.DataFrame(columns=REQUIRED_COLUMNS)
    with (
        patch.object(agent, "fetch_huggingface", return_value=agent._generate_synthetic(5)),
        patch.object(agent, "scrape_forum", return_value=empty),
        patch.object(agent, "fetch_rss", return_value=empty),
    ):
        # Redirect output paths
        agent._raw_path = tmp_path / "raw"
        agent._raw_path.mkdir()
        agent._reports_path = tmp_path / "reports"
        agent._reports_path.mkdir()
        agent.run()

    memory_file = tmp_path / "reports" / "context_memory.json"
    assert memory_file.exists(), "context_memory.json not created"
    with open(memory_file) as fh:
        data = json.load(fh)
    assert "metrics" in data
    assert "total_rows" in data["metrics"]


# ------------------------------------------------------------------ #
#  10. summary() contains no raw data                                  #
# ------------------------------------------------------------------ #

def test_summary_no_raw_data(agent: DataCollectionAgent, tmp_path: Path) -> None:
    """summary() must not contain 'data', 'rows', or 'texts' keys."""
    empty = pd.DataFrame(columns=REQUIRED_COLUMNS)
    with (
        patch.object(agent, "fetch_huggingface", return_value=agent._generate_synthetic(5)),
        patch.object(agent, "scrape_forum", return_value=empty),
        patch.object(agent, "fetch_rss", return_value=empty),
    ):
        agent._raw_path = tmp_path / "raw"
        agent._raw_path.mkdir()
        agent._reports_path = tmp_path / "reports"
        agent._reports_path.mkdir()
        agent.run()

    s = agent.summary()
    forbidden = {"data", "rows", "texts"}
    assert not forbidden.intersection(s.keys()), f"summary() contains forbidden keys: {forbidden.intersection(s.keys())}"


# ------------------------------------------------------------------ #
#  11. Explicit deduplication test                                     #
# ------------------------------------------------------------------ #

def test_merge_deduplication(agent: DataCollectionAgent) -> None:
    """merge() must collapse identical texts regardless of case/whitespace."""
    rows = [
        {"text": "The mainsail was reefed in heavy weather offshore",
         "label": "unlabeled", "source": "a", "collected_at": "2024-01-01"},
        {"text": "THE MAINSAIL WAS REEFED IN HEAVY WEATHER OFFSHORE",
         "label": "unlabeled", "source": "b", "collected_at": "2024-01-01"},
        {"text": "  the mainsail was reefed in heavy weather offshore  ",
         "label": "unlabeled", "source": "c", "collected_at": "2024-01-01"},
    ]
    result = agent.merge([pd.DataFrame(rows)])
    assert len(result) == 1


# ------------------------------------------------------------------ #
#  12. run() returns non-empty DataFrame                               #
# ------------------------------------------------------------------ #

def test_run_returns_dataframe(agent: DataCollectionAgent, tmp_path: Path) -> None:
    """run() must return a non-empty DataFrame with the required schema."""
    empty = pd.DataFrame(columns=REQUIRED_COLUMNS)
    with (
        patch.object(agent, "fetch_huggingface", return_value=agent._generate_synthetic(20)),
        patch.object(agent, "scrape_forum", return_value=empty),
        patch.object(agent, "fetch_rss", return_value=empty),
    ):
        agent._raw_path = tmp_path / "raw"
        agent._raw_path.mkdir()
        agent._reports_path = tmp_path / "reports"
        agent._reports_path.mkdir()
        result = agent.run()

    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0
    assert list(result.columns) == REQUIRED_COLUMNS


# ------------------------------------------------------------------ #
#  13. StackExchange graceful degradation                              #
# ------------------------------------------------------------------ #

def test_stackexchange_fetch_graceful(agent: DataCollectionAgent) -> None:
    """fetch_stackexchange() must not raise on network failure and return correct schema."""
    import requests as req_module
    with patch.object(req_module, "get", side_effect=req_module.RequestException("network error")):
        result = agent.fetch_stackexchange()
    assert isinstance(result, pd.DataFrame)
    # On error: empty DataFrame with base columns (id added later by merge)
    assert result.empty or set(["text", "label", "source", "collected_at"]).issubset(result.columns)


# ------------------------------------------------------------------ #
#  14. StackExchange calls _is_crawl_allowed first                    #
# ------------------------------------------------------------------ #

def test_stackexchange_robots_checked(agent: DataCollectionAgent) -> None:
    """fetch_stackexchange() must call _is_crawl_allowed() before any HTTP request."""
    calls: list[str] = []

    def _mock_crawl_allowed(url: str, ua: str) -> bool:
        calls.append(url)
        return False  # disallow — so no HTTP requests follow

    with patch.object(agent, "_is_crawl_allowed", side_effect=_mock_crawl_allowed):
        result = agent.fetch_stackexchange()

    assert len(calls) >= 1, "_is_crawl_allowed was never called"
    assert any("stackexchange" in c for c in calls), (
        f"Expected stackexchange URL in calls, got: {calls}"
    )
    assert result.empty, "Should return empty DataFrame when crawl is disallowed"


# ------------------------------------------------------------------ #
#  15. RSS handles 4 feeds without exception                          #
# ------------------------------------------------------------------ #

def test_rss_new_feeds(agent: DataCollectionAgent) -> None:
    """fetch_rss() must handle the full list of 4 configured feeds without raising."""
    import feedparser

    call_count = 0

    def _mock_parse(url: str, *args, **kwargs) -> MagicMock:
        nonlocal call_count
        call_count += 1
        mock = MagicMock()
        mock.bozo = False
        entry = MagicMock()
        entry.get = lambda k, d="": {
            "title": f"Test sailing article {call_count}",
            "summary": "A detailed summary about sailing and navigation techniques at sea.",
        }.get(k, d)
        mock.entries = [entry]
        return mock

    with patch.object(feedparser, "parse", side_effect=_mock_parse):
        result = agent.fetch_rss()

    # Config has 4 feeds — all should be called
    rss_feeds = agent._cfg["sources"]["rss"]["feeds"]
    assert call_count == len(rss_feeds), (
        f"Expected {len(rss_feeds)} feed calls, got {call_count}"
    )
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(rss_feeds), "Should have one entry per feed"


# ------------------------------------------------------------------ #
#  16. Kaggle graceful degradation                                    #
# ------------------------------------------------------------------ #

def test_kaggle_fetch_graceful(agent: DataCollectionAgent, monkeypatch: pytest.MonkeyPatch) -> None:
    """fetch_kaggle() must gracefully skip when KAGGLE_API_TOKEN is absent."""
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    result = agent.fetch_kaggle()
    assert isinstance(result, pd.DataFrame)
    assert result.empty
    assert list(result.columns) == REQUIRED_COLUMNS
