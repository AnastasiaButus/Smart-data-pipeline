"""Programmatic EDA export for step 1.4."""

from __future__ import annotations

import base64
from html import escape
import json
import re
import sys
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import yaml
from loguru import logger
from wordcloud import STOPWORDS as WORDCLOUD_STOPWORDS
from wordcloud import WordCloud

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.llm_client import GeminiLLMClient

REPORTS_DIR = ROOT / "reports"
DATASET_PATH = ROOT / "data" / "raw" / "dataset.parquet"
CLEAN_DATASET_PATH = ROOT / "data" / "raw" / "dataset_clean.parquet"
DOMAIN_SPEC_PATH = ROOT / "reports" / "domain_spec.json"
WORDCLOUD_ALL_PATH = REPORTS_DIR / "wordcloud_all.png"
WORDCLOUD_DOMAIN_PATH = REPORTS_DIR / "wordcloud_domain.png"
HYPOTHESES_PATH = REPORTS_DIR / "eda_hypotheses.json"
EDA_REPORT_PATH = REPORTS_DIR / "eda_report.html"
EDA_METADATA_PATH = REPORTS_DIR / "eda_metadata.json"

THEMATIC_SOURCE_NAMES = {"stackexchange_sailing", "sailingforums"}
BASE_STOPWORDS = {
    "wp",
    "p",
    "content",
    "wp-content",
    "wp-content-uploads",
    "uploads",
    "timeincuk",
    "inspirewp",
    "attachment",
    "attachment-medium",
    "medium",
    "size-medium",
    "size-full",
    "height",
    "width",
    "figcaption",
    "figure",
    "keyassets",
    "href",
    "nofollow",
    "www",
    "http",
    "https",
    "com",
    "html",
    "class",
    "entry",
    "entry-lead-paragraph",
    "lead",
    "paragraph",
    "strong",
    "net",
    "appeared",
    "first",
    "alt",
    "jpg",
    "png",
    "style",
    "margin",
    "bottom",
    "display",
    "block",
    "one",
    "get",
    "got",
    "even",
    "also",
    "well",
    "new",
    "year",
    "time",
    "way",
    "make",
    "made",
    "said",
    "say",
    "go",
    "going",
    "still",
    "yachtingworld",
    "cruisingworld",
    "sailmagazine",
    "48north",
    "sailingforums",
    "continue",
    "reading",
    "post",
    "image",
    "photo",
    "live",
    "site",
    "amp",
    "img",
    "src",
    "rel",
}
ENGLISH_STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "ain",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "aren",
    "aren't",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "couldn",
    "couldn't",
    "d",
    "did",
    "didn",
    "didn't",
    "do",
    "does",
    "doesn",
    "doesn't",
    "doing",
    "don",
    "don't",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "hadn",
    "hadn't",
    "has",
    "hasn",
    "hasn't",
    "have",
    "haven",
    "haven't",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "isn",
    "isn't",
    "it",
    "it's",
    "its",
    "itself",
    "just",
    "ll",
    "m",
    "ma",
    "me",
    "mightn",
    "mightn't",
    "more",
    "most",
    "mustn",
    "mustn't",
    "my",
    "myself",
    "needn",
    "needn't",
    "no",
    "nor",
    "not",
    "now",
    "o",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "re",
    "s",
    "same",
    "shan",
    "shan't",
    "she",
    "she's",
    "should",
    "should've",
    "shouldn",
    "shouldn't",
    "so",
    "some",
    "such",
    "t",
    "than",
    "that",
    "that'll",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "ve",
    "very",
    "was",
    "wasn",
    "wasn't",
    "we",
    "were",
    "weren",
    "weren't",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "won",
    "won't",
    "wouldn",
    "wouldn't",
    "y",
    "you",
    "you'd",
    "you'll",
    "you're",
    "you've",
    "your",
    "yours",
    "yourself",
    "yourselves",
}
BOOTSTRAP_META_STOPWORDS = {
    "guide",
    "guides",
    "overview",
    "tutorial",
    "training",
    "note",
    "notes",
    "community",
    "members",
    "discuss",
    "thread",
    "threads",
    "example",
    "examples",
    "practical",
    "concrete",
    "detailed",
    "sections",
    "recognised",
    "recognize",
    "reliable",
    "uncertain",
    "explicit",
    "implicit",
    "summarise",
    "summary",
    "analyst",
    "reviews",
    "highlights",
    "broader",
    "descriptions",
    "partial",
    "appears",
    "late",
    "article",
    "articles",
    "cross",
    "functional",
    "cross-functional",
    "learning",
    "resources",
    "best",
    "practice",
    "practices",
    "quality",
    "review",
    "checklists",
    "begins",
    "belongs",
    "mentions",
    "touches",
    "present",
    "context",
    "compare",
    "compares",
    "approaches",
    "workflow",
    "workflows",
}

HTML_STYLE = """
<style>
  body {
    font-family: -apple-system, BlinkMacSystemFont,
                 'Segoe UI', sans-serif;
    max-width: 1100px;
    margin: 0 auto;
    padding: 32px 24px;
    background: #f8f9fa;
    color: #1a1a2e;
  }
  h1 {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 4px;
    color: #0f3460;
  }
  h2 {
    font-size: 1.3rem;
    font-weight: 600;
    margin-top: 48px;
    margin-bottom: 8px;
    padding-bottom: 8px;
    border-bottom: 2px solid #0f3460;
    color: #0f3460;
    cursor: pointer;
    user-select: none;
  }
  h2::after { content: " ▾"; font-size: 0.8em; color: #999; }
  h2.collapsed::after { content: " ▸"; }
  .subtitle {
    color: #666;
    margin-bottom: 32px;
    font-size: 0.95rem;
  }
  .chart-block {
    background: white;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
  }
  .insight {
    background: #eef4ff;
    border-left: 4px solid #0f3460;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin-top: 12px;
    font-size: 0.9rem;
    color: #333;
    line-height: 1.6;
  }
  .insight strong { color: #0f3460; }
  .hypothesis-list {
    list-style: none;
    padding: 0;
  }
  .hypothesis-list li {
    background: white;
    border-left: 4px solid #16a085;
    padding: 14px 18px;
    margin-bottom: 10px;
    border-radius: 0 8px 8px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    font-size: 0.92rem;
    line-height: 1.6;
  }
  .hypothesis-list li::before {
    content: "💡 ";
  }
  .stat-badge {
    display: inline-block;
    background: #0f3460;
    color: white;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.8rem;
    margin-right: 6px;
  }
  .conclusions {
    background: white;
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
  }
  .conclusions ul {
    line-height: 1.9;
    color: #444;
  }
  .section-content { transition: opacity 0.2s; }
  .section-content.hidden { display: none; }
</style>
"""

COLLAPSIBLE_SCRIPT = """
<script>
document.querySelectorAll('h2').forEach(h2 => {
  h2.addEventListener('click', () => {
    h2.classList.toggle('collapsed');
    let next = h2.nextElementSibling;
    while (next && next.tagName !== 'H2') {
      next.classList.toggle('hidden');
      next = next.nextElementSibling;
    }
  });
});
</script>
"""


def load_inputs(project_root: Path | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the dataset and domain spec from the project workspace."""
    root = project_root or ROOT
    clean_dataset_path = root / "data" / "raw" / "dataset_clean.parquet"
    raw_dataset_path = root / "data" / "raw" / "dataset.parquet"
    dataset_path = clean_dataset_path if clean_dataset_path.exists() else raw_dataset_path
    domain_spec_path = root / "reports" / "domain_spec.json"

    df = pd.read_parquet(dataset_path)
    spec = json.loads(domain_spec_path.read_text(encoding="utf-8"))

    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str)
    df["source"] = df["source"].fillna("unknown").astype(str)
    df["text_len"] = df["text"].str.len()
    logger.info(
        "EDA inputs loaded from {}: rows={}, sources={}",
        dataset_path,
        len(df),
        df["source"].nunique(),
    )
    return df, spec


def load_topic(project_root: Path | None = None) -> str:
    """Load the current domain topic from ``config.yaml``."""
    root = project_root or ROOT
    config = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8")) or {}
    return str(config.get("domain", {}).get("topic", "sailing and yacht navigation"))


def normalize_topic_name(topic: str) -> str:
    """Normalize topic text for comparisons and metadata."""
    return " ".join(str(topic).strip().lower().split())


def build_stopwords(
    domain_spec: dict[str, Any],
    topic: str | None = None,
    llm_client: GeminiLLMClient | None = None,
    project_root: Path | None = None,
) -> set[str]:
    """Build stopwords for corpus analysis and WordCloud generation."""
    root = project_root or ROOT
    current_topic = topic or load_topic(root)
    config = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8")) or {}
    client = llm_client or GeminiLLMClient(config_path=str(root / "config.yaml"))

    base_stopwords = set(BASE_STOPWORDS)
    final_stopwords = client.generate_stopwords(current_topic, base_stopwords)
    stopwords = (
        set(ENGLISH_STOPWORDS)
        | set(WORDCLOUD_STOPWORDS)
        | set(BOOTSTRAP_META_STOPWORDS)
        | final_stopwords
    )
    topic_tokens = {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", str(current_topic).lower())
        if len(token) >= 3
    }
    stopwords |= topic_tokens

    current_classes = [
        str(item).strip().lower()
        for item in config.get("domain", {}).get("classes", [])
        if str(item).strip()
    ]
    normalized_topic = "_".join(str(current_topic).strip().lower().split())
    fallback_suffixes = {"basics", "tools", "workflows", "issues", "advanced"}
    for class_name in current_classes:
        if class_name.startswith(f"{normalized_topic}_"):
            suffix = class_name.split("_", 1)[1]
            if suffix in fallback_suffixes:
                stopwords.add(suffix)
                stopwords.update(suffix.split("_"))

    keyword_tokens = {
        token
        for keywords in domain_spec.get("keywords_by_class", {}).values()
        for keyword in keywords
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", str(keyword).lower())
        if token in stopwords or token in final_stopwords
    }
    return stopwords | keyword_tokens


def is_thematic_source(source_name: str) -> bool:
    """Return ``True`` for domain-oriented sources."""
    source = str(source_name or "").strip().lower()
    if not source or source == "unknown":
        return False
    if source.startswith("huggingface_"):
        return False
    if source.startswith(("rss_", "stackexchange_", "topic_bootstrap_", "kaggle_")):
        return True
    if "forum" in source:
        return True
    return source in THEMATIC_SOURCE_NAMES


def tokenize_text(text: str, stopwords: set[str]) -> list[str]:
    """Tokenize text with lightweight regex cleaning."""
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower())
    return [token for token in tokens if token not in stopwords]


def compute_thematic_stats(df: pd.DataFrame) -> dict[str, float]:
    """Compute thematic versus non-thematic row shares."""
    thematic_mask = df["source"].apply(is_thematic_source)
    thematic_rows = int(thematic_mask.sum())
    total_rows = int(len(df))
    off_topic_rows = total_rows - thematic_rows
    thematic_pct = round((thematic_rows / total_rows) * 100, 2) if total_rows else 0.0
    off_topic_pct = round((off_topic_rows / total_rows) * 100, 2) if total_rows else 0.0
    return {
        "thematic_rows": thematic_rows,
        "off_topic_rows": off_topic_rows,
        "thematic_pct": thematic_pct,
        "off_topic_pct": off_topic_pct,
    }


def source_distribution_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return sorted source counts for plotting."""
    source_df = (
        df["source"]
        .value_counts()
        .rename_axis("source")
        .reset_index(name="count")
        .sort_values("count", ascending=True)
    )
    return source_df


def build_rows_per_source_figure(
    source_df: pd.DataFrame,
    thematic_stats: dict[str, float],
) -> go.Figure:
    """Build the rows-per-source bar chart."""
    fig = px.bar(
        source_df,
        x="count",
        y="source",
        orientation="h",
        color="source",
        title="Rows per source",
        hover_data={"count": True, "source": True},
    )
    fig.update_layout(showlegend=False, height=460)
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=1.0,
        y=1.08,
        text=(
            f"Thematic rows: {thematic_stats['thematic_pct']}% | "
            f"Non-thematic rows: {thematic_stats['off_topic_pct']}%"
        ),
        showarrow=False,
    )
    return fig


def build_source_pie_figure(
    source_df: pd.DataFrame,
    thematic_stats: dict[str, float],
) -> go.Figure:
    """Build the source share pie chart."""
    fig = px.pie(
        source_df,
        values="count",
        names="source",
        title="Source share (%)",
        hover_data={"count": True},
    )
    fig.update_traces(textinfo="percent+label", hovertemplate="%{label}: %{percent}<extra></extra>")
    fig.add_annotation(
        x=0.5,
        y=-0.12,
        xref="paper",
        yref="paper",
        showarrow=False,
        text=f"Non-thematic share visible in HF-heavy sources: {thematic_stats['off_topic_pct']}%",
    )
    return fig


def build_text_length_histogram(df: pd.DataFrame) -> go.Figure:
    """Build the text length histogram with summary lines."""
    median_len = float(df["text_len"].median())
    mean_len = float(df["text_len"].mean())
    p95_len = float(df["text_len"].quantile(0.95))

    fig = px.histogram(
        df,
        x="text_len",
        nbins=50,
        title="Text length distribution",
    )
    for value, label, color in [
        (median_len, "median", "#1f77b4"),
        (mean_len, "mean", "#2ca02c"),
        (p95_len, "p95", "#d62728"),
    ]:
        fig.add_vline(x=value, line_dash="dash", line_color=color)
        fig.add_annotation(
            x=value,
            y=0.98,
            yref="paper",
            text=f"{label}: {value:.1f}",
            showarrow=False,
            bgcolor="white",
            bordercolor=color,
        )
    return fig


def build_source_boxplot(df: pd.DataFrame) -> go.Figure:
    """Build the horizontal boxplot of text length by source."""
    fig = px.box(
        df,
        x="text_len",
        y="source",
        orientation="h",
        points="outliers",
        title="Text length by source",
        color="source",
    )
    fig.update_layout(showlegend=False, height=500)
    return fig


def save_wordcloud_image(
    texts: list[str],
    stopwords: set[str],
    output_path: Path,
    colormap: str = "Blues",
) -> str:
    """Create and save a WordCloud image, returning a base64 HTML payload."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    token_counter: Counter[str] = Counter()
    for text in texts:
        token_counter.update(tokenize_text(str(text), stopwords))
    if not token_counter:
        token_counter.update(
            ["dataset", "analysis", "classification", "review", "quality"]
        )
    cloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        min_font_size=10,
        stopwords=stopwords,
        colormap=colormap,
        collocations=False,
    ).generate_from_frequencies(token_counter)

    plt.figure(figsize=(10, 5))
    plt.imshow(cloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    buffer = BytesIO()
    cloud.to_image().save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def compute_quality_preview(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute quality preview metrics for DataQualityAgent planning."""
    records: list[dict[str, Any]] = []
    for source_name, group in df.groupby("source", sort=True):
        text_norm = group["text"].str.strip().str.lower()
        duplicates = int(text_norm.duplicated().sum())
        html_hits = group["text"].str.contains(
            r"&(?:[a-zA-Z]+|#[0-9]+|#x[0-9a-fA-F]+);|href|nofollow|www",
            case=False,
            regex=True,
        )

        records.append(
            {
                "source": source_name,
                "count": int(len(group)),
                "avg_len": round(float(group["text_len"].mean()), 2),
                "min_len": int(group["text_len"].min()),
                "max_len": int(group["text_len"].max()),
                "empty_texts": round(float((group["text"].str.strip() == "").mean() * 100), 2),
                "short_texts(<50)": round(float((group["text_len"] < 50).mean() * 100), 2),
                "long_texts(>1000)": round(float((group["text_len"] > 1000).mean() * 100), 2),
                "html_entities": round(float(html_hits.mean() * 100), 2),
                "duplicates_pct": round((duplicates / len(group)) * 100, 2) if len(group) else 0.0,
                "html_entity_pct": round(float(html_hits.mean() * 100), 2),
                "short_text_pct": round(float((group["text_len"] < 50).mean() * 100), 2),
            }
        )

    quality_df = pd.DataFrame(records).sort_values("count", ascending=False)
    table_df = quality_df[
        [
            "source",
            "count",
            "avg_len",
            "min_len",
            "max_len",
            "html_entity_pct",
            "short_text_pct",
        ]
    ].copy()
    return quality_df, table_df


def build_quality_heatmap(quality_df: pd.DataFrame) -> go.Figure:
    """Build the source-by-problem heatmap."""
    columns = [
        "empty_texts",
        "short_texts(<50)",
        "long_texts(>1000)",
        "html_entities",
        "duplicates_pct",
    ]
    if quality_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Данных для quality heatmap пока нет",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 16},
        )
        fig.update_layout(title="Data quality preview by source", height=460)
        return fig

    z_values = quality_df[columns].fillna(0.0).values
    max_value = float(quality_df[columns].fillna(0.0).to_numpy().max()) if len(quality_df) else 0.0
    text_values = [[f"{float(value):.1f}" for value in row] for row in z_values]
    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=columns,
            y=quality_df["source"],
            text=text_values,
            texttemplate="%{text}",
            textfont={"size": 12},
            zmin=0,
            zmax=max(max_value, 1.0),
            colorscale="Blues" if max_value <= 0 else "RdYlGn_r",
            colorbar={"title": "Severity %"},
            hovertemplate="Source: %{y}<br>Problem: %{x}<br>Severity: %{z:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(title="Data quality preview by source", height=460)
    if max_value <= 0:
        fig.add_annotation(
            text="На текущих данных явных проблем качества не обнаружено",
            x=0.5,
            y=1.08,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 13},
        )
    return fig


def build_source_stats_table(table_df: pd.DataFrame) -> go.Figure:
    """Build the Plotly table with per-source stats."""
    fig = go.Figure(
        data=[
            go.Table(
                header={
                    "values": list(table_df.columns),
                    "fill_color": "#264653",
                    "font": {"color": "white"},
                    "align": "left",
                },
                cells={
                    "values": [table_df[column] for column in table_df.columns],
                    "fill_color": "#f1faee",
                    "align": "left",
                },
            )
        ]
    )
    fig.update_layout(title="Per-source summary statistics", height=380)
    return fig


def build_top_words_figure(df: pd.DataFrame, stopwords: set[str]) -> go.Figure:
    """Build a source-switchable top words chart."""
    fig = go.Figure()
    buttons: list[dict[str, Any]] = []

    sources = sorted(df["source"].unique())
    for index, source_name in enumerate(sources):
        tokens = [
            token
            for text in df.loc[df["source"] == source_name, "text"]
            for token in tokenize_text(text, stopwords)
        ]
        top_items = Counter(tokens).most_common(10)
        words = [item[0] for item in top_items]
        counts = [item[1] for item in top_items]

        fig.add_trace(
            go.Bar(
                x=words,
                y=counts,
                name=source_name,
                visible=index == 0,
            )
        )
        visible = [False] * len(sources)
        visible[index] = True
        buttons.append(
            {
                "label": source_name,
                "method": "update",
                "args": [
                    {"visible": visible},
                    {"title": f"Top 10 words - {source_name}"},
                ],
            }
        )

    fig.update_layout(
        title=f"Top 10 words - {sources[0] if sources else 'source'}",
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 1.02,
                "y": 1.15,
                "showactive": True,
            }
        ],
        xaxis_title="Word",
        yaxis_title="Count",
        height=420,
    )
    return fig


def save_hypotheses(
    dataset_summary: dict[str, Any],
    project_root: Path | None = None,
) -> list[str]:
    """Generate and persist EDA hypotheses."""
    root = project_root or ROOT
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    client = GeminiLLMClient(config_path=str(root / "config.yaml"))
    hypotheses = client.generate_eda_hypotheses(dataset_summary)
    (reports_dir / "eda_hypotheses.json").write_text(
        json.dumps(hypotheses, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("EDA hypotheses saved to {}", reports_dir / "eda_hypotheses.json")
    return hypotheses


def top_terms_from_texts(
    texts: list[str],
    stopwords: set[str],
    limit: int = 5,
) -> list[str]:
    """Return the most frequent informative tokens from the provided texts."""
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize_text(str(text), stopwords))
    return [token for token, _ in counter.most_common(limit)]


def summarize_quality_issues(quality_df: pd.DataFrame, limit: int = 3) -> list[str]:
    """Summarize the most visible quality issues across sources."""
    if quality_df.empty:
        return []

    issue_definitions = [
        ("html_entities", "HTML-артефакты"),
        ("short_texts(<50)", "короткие тексты"),
        ("long_texts(>1000)", "слишком длинные тексты"),
        ("duplicates_pct", "дубликаты"),
    ]
    issues: list[tuple[float, str]] = []
    for _, row in quality_df.iterrows():
        source_name = str(row.get("source", "unknown"))
        for column_name, label in issue_definitions:
            value = float(row.get(column_name, 0.0) or 0.0)
            if value > 0:
                issues.append((value, f"{label} в `{source_name}` ({value:.1f}%)"))
    issues.sort(key=lambda item: item[0], reverse=True)
    return [text for _, text in issues[:limit]]


def save_eda_metadata(
    topic: str,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Persist lightweight metadata describing which topic/data the EDA report belongs to."""
    root = project_root or ROOT
    raw_path = root / "data" / "raw" / "dataset.parquet"
    clean_path = root / "data" / "raw" / "dataset_clean.parquet"
    annotated_path = root / "data" / "labeled" / "annotated.parquet"
    metadata = {
        "topic": topic,
        "normalized_topic": normalize_topic_name(topic),
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "raw_dataset_mtime": raw_path.stat().st_mtime if raw_path.exists() else 0.0,
        "clean_dataset_mtime": clean_path.stat().st_mtime if clean_path.exists() else 0.0,
        "annotated_dataset_mtime": annotated_path.stat().st_mtime if annotated_path.exists() else 0.0,
    }
    EDA_METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("EDA metadata saved to {}", EDA_METADATA_PATH)
    return metadata


def build_conclusions_html(
    topic: str,
    thematic_stats: dict[str, float],
    quality_df: pd.DataFrame,
    source_df: pd.DataFrame,
) -> str:
    """Build the conclusions and recommendations section as styled HTML."""
    dominant_sources = ", ".join(
        source_df.sort_values("count", ascending=False)["source"].head(3).astype(str).tolist()
    ) or "нет данных"
    issues = summarize_quality_issues(quality_df)
    issue_html = "".join(f"<li>{escape(item)}</li>" for item in issues) or (
        "<li>Серьёзных проблем качества не обнаружено, но перед авторазметкой всё равно стоит "
        "проверить короткие тексты и шумные источники.</li>"
    )
    return f"""
<div class="conclusions section-content">
  <h2>Выводы и рекомендации</h2>
  <ul>
    <li>Текущая тема: <strong>{escape(topic)}</strong>.</li>
    <li>Тематических строк: <strong>{thematic_stats['thematic_rows']} ({thematic_stats['thematic_pct']:.1f}%)</strong>
        — нетематических: <strong>{thematic_stats['off_topic_rows']} ({thematic_stats['off_topic_pct']:.1f}%)</strong></li>
    <li>Доминирующие источники: <strong>{escape(dominant_sources)}</strong>.</li>
    {issue_html}
    <li><strong>DataQualityAgent</strong>: приоритет —
        очистка шумных источников, near-duplicate контроль и фильтрация слишком коротких текстов.</li>
    <li><strong>AnnotationAgent</strong>: сначала отделить доменный контент от общего шума,
        затем уточнять классы внутри тематических источников и отправлять спорные случаи в HITL.</li>
  </ul>
</div>
""".strip()


def build_insights(
    df: pd.DataFrame,
    source_df: pd.DataFrame,
    thematic_stats: dict[str, float],
    quality_df: pd.DataFrame,
    stopwords: set[str],
    topic: str,
) -> dict[str, str]:
    """Build dynamic insights for each EDA block."""
    total = int(len(df))
    n_sources = int(df["source"].nunique())
    hf_rows = int(
        source_df.loc[source_df["source"].str.startswith("huggingface_"), "count"].sum()
    )
    hf_pct = round((hf_rows / total) * 100, 1) if total else 0.0
    theme_pct = float(thematic_stats["thematic_pct"])
    domain_pct = float(thematic_stats["thematic_pct"])
    dominant_sources = source_df.sort_values("count", ascending=False).head(3)
    dominant_source_names = ", ".join(dominant_sources["source"].astype(str).tolist()) or "нет данных"
    bootstrap_rows = int(
        source_df.loc[source_df["source"].str.startswith("topic_bootstrap_"), "count"].sum()
    )
    bootstrap_pct = round((bootstrap_rows / total) * 100, 1) if total else 0.0
    bootstrap_note = (
        " Сейчас датасет почти полностью состоит из synthetic bootstrap-текстов, поэтому "
        "WordCloud показывает скорее шаблонные сигналы и учебные маркеры, чем лексику реальных внешних источников."
        if bootstrap_pct >= 70.0
        else ""
    )
    top_source = str(dominant_sources.iloc[0]["source"]) if not dominant_sources.empty else "нет данных"
    top_source_share = (
        round(float(dominant_sources.iloc[0]["count"]) / total * 100, 1)
        if total and not dominant_sources.empty
        else 0.0
    )
    all_terms = ", ".join(top_terms_from_texts(df["text"].tolist(), stopwords, limit=5)) or "ключевые термины темы"
    thematic_texts = df.loc[df["source"].apply(is_thematic_source), "text"].tolist()
    domain_terms = ", ".join(top_terms_from_texts(thematic_texts, stopwords, limit=5)) or "доменные термины"
    issues = summarize_quality_issues(quality_df, limit=2)
    issues_text = "; ".join(issues) if issues else "критичных аномалий по качеству почти нет"

    median_len = float(df["text_len"].median()) if total else 0.0
    mean_len = float(df["text_len"].mean()) if total else 0.0
    p95 = float(df["text_len"].quantile(0.95)) if total else 0.0
    longest_source = (
        df.groupby("source")["text_len"].mean().sort_values(ascending=False).index[0]
        if total
        else "нет данных"
    )
    max_source_len = (
        int(df.groupby("source")["text_len"].max().sort_values(ascending=False).iloc[0])
        if total
        else 0
    )

    rss_html = float(
        quality_df.loc[
            quality_df["source"].str.startswith("rss_"),
            "html_entities",
        ].max()
    ) if quality_df["source"].str.startswith("rss_").any() else 0.0

    return {
        "source_bar": (
            f"<strong>Наблюдение:</strong> Датасет содержит {total} строк из {n_sources} "
            f"источников. Крупнейшие источники: {escape(dominant_source_names)}. "
            f"Тематических данных {theme_pct:.1f}% — это важно учесть при аннотации по теме <strong>{escape(topic)}</strong>."
        ),
        "source_pie": (
            f"<strong>Наблюдение:</strong> Крупнейший источник сейчас — <strong>{escape(top_source)}</strong> "
            f"с долей {top_source_share:.1f}%. HuggingFace-источники занимают {hf_pct:.1f}% объёма, "
            f"а доменные данные составляют {domain_pct:.1f}%."
        ),
        "length_hist": (
            f"<strong>Наблюдение:</strong> Медиана длины текста — {median_len:.0f} символов, "
            f"среднее — {mean_len:.0f}. 95-й перцентиль: {p95:.0f} символов. "
            "Большинство текстов короткие — подходят для zero-shot классификации."
        ),
        "length_box": (
            f"<strong>Наблюдение:</strong> Источник <strong>{escape(str(longest_source))}</strong> "
            f"содержит самые длинные тексты (до {max_source_len} символов). "
            "Это хороший кандидат для дополнительной очистки и усечения."
        ),
        "wordcloud_all": (
            f"<strong>Наблюдение:</strong> В общем облаке чаще всего встречаются термины: "
            f"{escape(all_terms)}. Это помогает быстро проверить, что словарь соответствует теме "
            f"<strong>{escape(topic)}</strong>.{bootstrap_note}"
        ),
        "wordcloud_domain": (
            f"<strong>Наблюдение:</strong> В тематическом облаке доминируют: {escape(domain_terms)}. "
            "Если здесь появляются слова из прошлой темы, отчёт нужно пересобрать на свежих артефактах."
            f"{bootstrap_note}"
        ),
        "quality_heatmap": (
            f"<strong>Наблюдение:</strong> Самые заметные проблемы качества: {escape(issues_text)}. "
            f"HTML-артефакты в RSS-источниках достигают {rss_html:.0f}% и должны быть очищены до аннотации."
        ),
        "top_words": (
            "<strong>Наблюдение:</strong> Выбери источник в выпадающем списке и сравни лексику. "
            "Так проще увидеть, какие источники реально соответствуют теме, а какие добавляют общий шум."
        ),
    }


def build_hypotheses_html(hypotheses: list[str]) -> str:
    """Render hypotheses as a styled HTML list."""
    items = "".join(f"<li>{escape(item)}</li>" for item in hypotheses)
    return f"<ul class='hypothesis-list section-content'>{items}</ul>"


def wrap_chart_block(content_html: str, insight_html: str = "") -> str:
    """Wrap a chart or image block with optional insight copy."""
    insight = f"<div class='insight'>{insight_html}</div>" if insight_html else ""
    return f"<div class='chart-block section-content'>{content_html}{insight}</div>"


def build_eda_assets(project_root: Path | None = None) -> dict[str, Any]:
    """Run all EDA computations and return reusable assets."""
    root = project_root or ROOT
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    df, domain_spec = load_inputs(root)
    topic = load_topic(root)
    client = GeminiLLMClient(config_path=str(root / "config.yaml"))
    stopwords = build_stopwords(domain_spec, topic=topic, llm_client=client, project_root=root)
    thematic_stats = compute_thematic_stats(df)
    source_df = source_distribution_frame(df)

    fig_source_bar = build_rows_per_source_figure(source_df, thematic_stats)
    fig_source_pie = build_source_pie_figure(source_df, thematic_stats)
    fig_hist = build_text_length_histogram(df)
    fig_box = build_source_boxplot(df)

    all_wc_b64 = save_wordcloud_image(
        df["text"].tolist(),
        stopwords,
        reports_dir / "wordcloud_all.png",
        colormap="Blues",
    )
    domain_texts = df.loc[df["source"].apply(is_thematic_source), "text"].tolist()
    domain_wc_b64 = save_wordcloud_image(
        domain_texts,
        stopwords,
        reports_dir / "wordcloud_domain.png",
        colormap="viridis",
    )

    quality_df, table_df = compute_quality_preview(df)
    fig_heatmap = build_quality_heatmap(quality_df)
    fig_table = build_source_stats_table(table_df)
    fig_top_words = build_top_words_figure(df, stopwords)

    dataset_summary = client.build_dataset_summary(df)
    hypotheses = save_hypotheses(dataset_summary, root)
    conclusions_html = build_conclusions_html(topic, thematic_stats, quality_df, source_df)
    insights = build_insights(df, source_df, thematic_stats, quality_df, stopwords, topic)

    return {
        "df": df,
        "domain_spec": domain_spec,
        "topic": topic,
        "dataset_summary": dataset_summary,
        "thematic_stats": thematic_stats,
        "quality_df": quality_df,
        "table_df": table_df,
        "insights": insights,
        "figures": {
            "source_bar": fig_source_bar,
            "source_pie": fig_source_pie,
            "length_hist": fig_hist,
            "length_box": fig_box,
            "quality_heatmap": fig_heatmap,
            "source_table": fig_table,
            "top_words": fig_top_words,
        },
        "wordclouds": {
            "all_base64": all_wc_b64,
            "domain_base64": domain_wc_b64,
            "all_path": reports_dir / "wordcloud_all.png",
            "domain_path": reports_dir / "wordcloud_domain.png",
        },
        "hypotheses": hypotheses,
        "conclusions_html": conclusions_html,
    }


def _figure_to_html(fig: go.Figure, include_plotlyjs: str | bool) -> str:
    """Render a Plotly figure as embeddable HTML."""
    rendered = go.Figure(fig)
    rendered.update_layout(
        autosize=True,
        margin=dict(l=40, r=40, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(240,244,255,0.5)",
        font=dict(
            family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
            size=13,
        ),
        hoverlabel=dict(bgcolor="white", font_size=13),
    )
    return pio.to_html(
        rendered,
        full_html=False,
        include_plotlyjs=include_plotlyjs,
        config={
            "responsive": True,
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )


def export_eda_report(
    project_root: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    """Export the full interactive EDA HTML report."""
    root = project_root or ROOT
    report_path = output_path or (root / "reports" / "eda_report.html")
    assets = build_eda_assets(root)

    sections: list[str] = [
        "<html><head><meta charset='utf-8'><title>EDA Report</title>",
        HTML_STYLE,
        "</head><body>",
        "<h1>EDA Report</h1>",
        (
            "<div class='subtitle'>Interactive EDA for the current smart-data-pipeline dataset. "
            f"Текущая тема: <strong>{escape(str(assets['topic']))}</strong>.</div>"
        ),
        (
            "<div class='subtitle'>"
            f"<span class='stat-badge'>Rows: {assets['dataset_summary']['total_rows']}</span>"
            f"<span class='stat-badge'>Sources: {len(assets['dataset_summary']['source_distribution'])}</span>"
            f"<span class='stat-badge'>Thematic: {assets['thematic_stats']['thematic_pct']:.1f}%</span>"
            f"<span class='stat-badge'>Off-topic: {assets['thematic_stats']['off_topic_pct']:.1f}%</span>"
            "</div>"
        ),
        "<h2>Dataset overview</h2>",
        wrap_chart_block(
            _figure_to_html(assets["figures"]["source_bar"], include_plotlyjs="cdn"),
            assets["insights"]["source_bar"],
        ),
        wrap_chart_block(
            _figure_to_html(assets["figures"]["source_pie"], include_plotlyjs=False),
            assets["insights"]["source_pie"],
        ),
        "<h2>Text length</h2>",
        wrap_chart_block(
            _figure_to_html(assets["figures"]["length_hist"], include_plotlyjs=False),
            assets["insights"]["length_hist"],
        ),
        wrap_chart_block(
            _figure_to_html(assets["figures"]["length_box"], include_plotlyjs=False),
            assets["insights"]["length_box"],
        ),
        "<h2>WordCloud</h2>",
        wrap_chart_block(
            (
                "<a href='data:image/png;base64,"
                f"{assets['wordclouds']['all_base64']}' target='_blank'>"
                f"<img src='data:image/png;base64,{assets['wordclouds']['all_base64']}' "
                "style='max-width:100%;height:auto;border-radius:8px;cursor:zoom-in;' /></a>"
            ),
            assets["insights"]["wordcloud_all"],
        ),
        wrap_chart_block(
            (
                "<a href='data:image/png;base64,"
                f"{assets['wordclouds']['domain_base64']}' target='_blank'>"
                f"<img src='data:image/png;base64,{assets['wordclouds']['domain_base64']}' "
                "style='max-width:100%;height:auto;border-radius:8px;cursor:zoom-in;' /></a>"
            ),
            assets["insights"]["wordcloud_domain"],
        ),
        "<h2>Data quality preview</h2>",
        wrap_chart_block(
            _figure_to_html(assets["figures"]["quality_heatmap"], include_plotlyjs=False),
            assets["insights"]["quality_heatmap"],
        ),
        wrap_chart_block(
            _figure_to_html(assets["figures"]["source_table"], include_plotlyjs=False),
        ),
        "<h2>Top words by source</h2>",
        wrap_chart_block(
            _figure_to_html(assets["figures"]["top_words"], include_plotlyjs=False),
            assets["insights"]["top_words"],
        ),
        "<h2>LLM hypotheses</h2>",
        build_hypotheses_html(assets["hypotheses"]),
        assets["conclusions_html"],
        COLLAPSIBLE_SCRIPT,
        "</body></html>",
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(sections), encoding="utf-8")
    save_eda_metadata(str(assets["topic"]), root)
    logger.info("EDA report exported to {}", report_path)
    return report_path


def main() -> Path:
    """Run the EDA export using the project workspace."""
    return export_eda_report(ROOT, EDA_REPORT_PATH)


if __name__ == "__main__":
    main()
