"""Programmatic EDA export for step 1.4."""

from __future__ import annotations

import base64
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
from loguru import logger
from wordcloud import STOPWORDS as WORDCLOUD_STOPWORDS
from wordcloud import WordCloud

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.llm_client import GeminiLLMClient

REPORTS_DIR = ROOT / "reports"
DATASET_PATH = ROOT / "data" / "raw" / "dataset.parquet"
DOMAIN_SPEC_PATH = ROOT / "reports" / "domain_spec.json"
WORDCLOUD_ALL_PATH = REPORTS_DIR / "wordcloud_all.png"
WORDCLOUD_DOMAIN_PATH = REPORTS_DIR / "wordcloud_domain.png"
HYPOTHESES_PATH = REPORTS_DIR / "eda_hypotheses.json"
EDA_REPORT_PATH = REPORTS_DIR / "eda_report.html"

THEMATIC_SOURCE_NAMES = {"stackexchange_sailing", "sailingforums"}
CUSTOM_STOPWORDS = {
    "href",
    "nofollow",
    "www",
    "http",
    "https",
    "com",
    "html",
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


def load_inputs(project_root: Path | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the dataset and domain spec from the project workspace."""
    root = project_root or ROOT
    dataset_path = root / "data" / "raw" / "dataset.parquet"
    domain_spec_path = root / "reports" / "domain_spec.json"

    df = pd.read_parquet(dataset_path)
    spec = json.loads(domain_spec_path.read_text(encoding="utf-8"))

    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str)
    df["source"] = df["source"].fillna("unknown").astype(str)
    df["text_len"] = df["text"].str.len()
    logger.info("EDA inputs loaded: rows={}, sources={}", len(df), df["source"].nunique())
    return df, spec


def build_stopwords(domain_spec: dict[str, Any]) -> set[str]:
    """Build stopwords for corpus analysis and WordCloud generation."""
    stopwords = set(ENGLISH_STOPWORDS) | set(WORDCLOUD_STOPWORDS) | CUSTOM_STOPWORDS
    keyword_tokens = {
        token
        for keywords in domain_spec.get("keywords_by_class", {}).values()
        for keyword in keywords
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", str(keyword).lower())
        if token in stopwords or token in CUSTOM_STOPWORDS
    }
    return stopwords | keyword_tokens


def is_thematic_source(source_name: str) -> bool:
    """Return ``True`` for domain-oriented sources."""
    return source_name.startswith("rss_") or source_name in THEMATIC_SOURCE_NAMES


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
    corpus = " ".join(texts).strip() or "sailing navigation safety weather equipment"
    cloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        min_font_size=10,
        stopwords=stopwords,
        colormap=colormap,
    ).generate(corpus)

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
    fig = go.Figure(
        data=go.Heatmap(
            z=quality_df[columns].values,
            x=columns,
            y=quality_df["source"],
            colorscale="RdYlGn_r",
            colorbar={"title": "Severity %"},
            hovertemplate="Source: %{y}<br>Problem: %{x}<br>Severity: %{z:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(title="Data quality preview by source", height=460)
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


def build_conclusions_markdown(
    thematic_stats: dict[str, float],
    quality_df: pd.DataFrame,
) -> str:
    """Build the conclusions and recommendations section."""
    problem_columns = [
        "empty_texts",
        "short_texts(<50)",
        "long_texts(>1000)",
        "html_entities",
        "duplicates_pct",
    ]
    top_problems = (
        quality_df[problem_columns]
        .mean()
        .sort_values(ascending=False)
        .head(3)
        .index.tolist()
    )
    top_problem_text = ", ".join(top_problems)

    return (
        "## Conclusions and recommendations\n\n"
        f"- Thematic rows: {thematic_stats['thematic_rows']} ({thematic_stats['thematic_pct']}%).\n"
        f"- Non-thematic rows: {thematic_stats['off_topic_rows']} ({thematic_stats['off_topic_pct']}%).\n"
        f"- Top quality issues preview: {top_problem_text}.\n"
        "- DataQualityAgent should prioritize HTML cleanup, duplicate control, and short-text filtering.\n"
        "- AnnotationAgent should lean on `other_or_offtopic` for HF-heavy rows and start with thematic sources first.\n"
    )


def build_eda_assets(project_root: Path | None = None) -> dict[str, Any]:
    """Run all EDA computations and return reusable assets."""
    root = project_root or ROOT
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    df, domain_spec = load_inputs(root)
    stopwords = build_stopwords(domain_spec)
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

    client = GeminiLLMClient(config_path=str(root / "config.yaml"))
    dataset_summary = client.build_dataset_summary(df)
    hypotheses = save_hypotheses(dataset_summary, root)
    conclusions_md = build_conclusions_markdown(thematic_stats, quality_df)

    return {
        "df": df,
        "domain_spec": domain_spec,
        "dataset_summary": dataset_summary,
        "thematic_stats": thematic_stats,
        "quality_df": quality_df,
        "table_df": table_df,
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
        "conclusions_md": conclusions_md,
    }


def _figure_to_html(fig: go.Figure, include_plotlyjs: str | bool) -> str:
    """Render a Plotly figure as embeddable HTML."""
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=include_plotlyjs,
        config={"displaylogo": False},
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
        "<html><head><meta charset='utf-8'><title>EDA Report</title></head><body>",
        "<h1>EDA Report</h1>",
        "<p>Interactive EDA for the current smart-data-pipeline dataset.</p>",
        "<h2>Dataset overview</h2>",
        _figure_to_html(assets["figures"]["source_bar"], include_plotlyjs="cdn"),
        _figure_to_html(assets["figures"]["source_pie"], include_plotlyjs=False),
        "<h2>Text length</h2>",
        _figure_to_html(assets["figures"]["length_hist"], include_plotlyjs=False),
        _figure_to_html(assets["figures"]["length_box"], include_plotlyjs=False),
        "<h2>WordCloud - full corpus</h2>",
        (
            "<a href='data:image/png;base64,"
            f"{assets['wordclouds']['all_base64']}' target='_blank'>"
            f"<img src='data:image/png;base64,{assets['wordclouds']['all_base64']}' "
            "style='max-width:100%;height:auto;border:1px solid #ccc;' /></a>"
        ),
        "<h2>WordCloud - thematic sources</h2>",
        (
            "<a href='data:image/png;base64,"
            f"{assets['wordclouds']['domain_base64']}' target='_blank'>"
            f"<img src='data:image/png;base64,{assets['wordclouds']['domain_base64']}' "
            "style='max-width:100%;height:auto;border:1px solid #ccc;' /></a>"
        ),
        "<h2>Data quality preview</h2>",
        _figure_to_html(assets["figures"]["quality_heatmap"], include_plotlyjs=False),
        _figure_to_html(assets["figures"]["source_table"], include_plotlyjs=False),
        "<h2>Top words by source</h2>",
        _figure_to_html(assets["figures"]["top_words"], include_plotlyjs=False),
        "<h2>LLM hypotheses</h2>",
        "<ul>" + "".join(f"<li>{item}</li>" for item in assets["hypotheses"]) + "</ul>",
        assets["conclusions_md"].replace("\n", "<br>"),
        "</body></html>",
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(sections), encoding="utf-8")
    logger.info("EDA report exported to {}", report_path)
    return report_path


def main() -> Path:
    """Run the EDA export using the project workspace."""
    return export_eda_report(ROOT, EDA_REPORT_PATH)


if __name__ == "__main__":
    main()
