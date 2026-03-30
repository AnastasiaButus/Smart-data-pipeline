"""Tests for EDA notebook export utilities."""

from __future__ import annotations

import json
import sys
from pathlib import Path

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
        "generate_eda_hypotheses",
        lambda self, dataset_summary: _mock_hypotheses(),
    )

    export_eda.export_eda_report(ROOT, ROOT / "reports" / "eda_report.html")

    hypotheses_path = ROOT / "reports" / "eda_hypotheses.json"
    assert hypotheses_path.exists()
    payload = json.loads(hypotheses_path.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert len(payload) >= 3
