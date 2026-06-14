"""Streamlit + Plotly dashboard for the Hacker News data lake."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Make project root importable so we can reuse `config`.
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

HN_ORANGE = "#ff6600"
PLOTLY_TEMPLATE = "plotly_white"

st.set_page_config(
    page_title="Hacker News Analytics",
    page_icon="📰",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    """Load the date-partitioned Parquet dataset (cached across reruns).

    Reading the dataset root reconstructs the ``ingestion_date`` partition
    column and concatenates every day's partition into one DataFrame.
    """
    df = pd.read_parquet(path)  # pyarrow discovers ingestion_date=*/ partitions
    # The partition key comes back as a categorical/string; normalise to str.
    if "ingestion_date" in df.columns:
        df["ingestion_date"] = df["ingestion_date"].astype(str)
    # Parquet round-trips published_at as tz-aware datetime already.
    return df


def kpi_row(df: pd.DataFrame) -> None:
    top_author = (
        df.groupby("author")["score"].sum().sort_values(ascending=False).index[0]
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stories", len(df))
    c2.metric("Avg. score", f"{df['score'].mean():.0f}")
    c3.metric("Total comments", f"{int(df['num_comments'].sum()):,}")
    c4.metric("Top author", top_author)


def chart_top_stories(df: pd.DataFrame) -> None:
    top = df.nlargest(15, "score").sort_values("score")
    top = top.assign(short_title=top["title"].str.slice(0, 60))
    fig = px.bar(
        top,
        x="score",
        y="short_title",
        orientation="h",
        text="score",
        color="score",
        color_continuous_scale="Oranges",
        labels={"short_title": "", "score": "Score"},
        title="Top 15 Stories by Score",
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        coloraxis_showscale=False,
        height=520,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)


def chart_score_distribution(df: pd.DataFrame) -> None:
    fig = px.histogram(
        df,
        x="score",
        nbins=15,
        color_discrete_sequence=[HN_ORANGE],
        labels={"score": "Score"},
        title="Score Distribution",
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=400,
        bargap=0.05,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def chart_top_authors(df: pd.DataFrame) -> None:
    authors = (
        df.groupby("author")
        .agg(total_score=("score", "sum"), stories=("id", "count"))
        .sort_values("total_score", ascending=False)
        .head(10)
        .reset_index()
        .sort_values("total_score")
    )
    fig = px.bar(
        authors,
        x="total_score",
        y="author",
        orientation="h",
        text="stories",
        color="total_score",
        color_continuous_scale="Oranges",
        labels={"total_score": "Total score", "author": ""},
        title="Top Authors (bar = total score, label = # stories)",
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        coloraxis_showscale=False,
        height=400,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def chart_score_vs_comments(df: pd.DataFrame) -> None:
    fig = px.scatter(
        df,
        x="score",
        y="num_comments",
        size="engagement",
        color="domain",
        hover_name="title",
        labels={"score": "Score", "num_comments": "Comments"},
        title="Score vs. Comments (bubble size = engagement)",
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=460,
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def chart_domains(df: pd.DataFrame) -> None:
    domains = (
        df["domain"].value_counts().head(10).rename_axis("domain").reset_index(name="count")
    ).sort_values("count")
    fig = px.bar(
        domains,
        x="count",
        y="domain",
        orientation="h",
        text="count",
        color="count",
        color_continuous_scale="Oranges",
        labels={"count": "Stories", "domain": ""},
        title="Top Source Domains",
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        coloraxis_showscale=False,
        height=400,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def story_table(df: pd.DataFrame) -> None:
    view = df[
        ["rank", "title", "author", "score", "num_comments", "domain",
         "published_date", "url"]
    ].copy()
    st.dataframe(
        view,
        use_container_width=True,
        hide_index=True,
        column_config={
            "rank": st.column_config.NumberColumn("Rank", width="small"),
            "title": st.column_config.TextColumn("Title", width="large"),
            "author": "Author",
            "score": st.column_config.NumberColumn("Score"),
            "num_comments": st.column_config.NumberColumn("Comments"),
            "domain": "Domain",
            "published_date": "Published",
            "url": st.column_config.LinkColumn("Link", display_text="open ↗"),
        },
    )


def main() -> None:
    st.markdown(
        f"<h1 style='color:{HN_ORANGE};margin-bottom:0'>📰 Hacker News Analytics</h1>"
        "<p style='color:gray;margin-top:4px'>Top 30 stories — live data lake (Parquet)</p>",
        unsafe_allow_html=True,
    )

    # The dataset is a partitioned directory; treat "no partitions yet" as empty.
    has_data = config.PARQUET_DATASET.exists() and any(
        config.PARQUET_DATASET.glob(f"{config.PARTITION_COL}=*")
    )
    if not has_data:
        st.error(
            "No data found. Run the pipeline first:\n\n"
            "```\npython -m src.run_pipeline\n```"
        )
        st.stop()

    df = load_data(str(config.PARQUET_DATASET))

    # --- Sidebar filters ---------------------------------------------------
    with st.sidebar:
        st.header("Filters")
        ingested = df["ingestion_date"].iloc[0]
        st.caption(f"Data ingested: **{ingested}**")

        # Keyword search across title / author / domain (case-insensitive).
        keyword = st.text_input(
            "Keyword",
            placeholder="Search title, author or domain…",
        ).strip()

        # Publication date range (based on each story's publish time).
        pub_dates = df["published_at"].dt.date
        min_date, max_date = pub_dates.min(), pub_dates.max()
        date_range = st.date_input(
            "Published between",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        # Minimum score.
        min_score, max_score = int(df["score"].min()), int(df["score"].max())
        if min_score == max_score:
            max_score += 1  # avoid an invalid slider range
        threshold = st.slider("Minimum score", min_score, max_score, min_score)

        st.caption("Re-run `python -m src.run_pipeline` to refresh the data.")

    # --- Apply filters -----------------------------------------------------
    filtered = df.copy()

    if keyword:
        mask = (
            filtered["title"].str.contains(keyword, case=False, na=False, regex=False)
            | filtered["author"].str.contains(keyword, case=False, na=False, regex=False)
            | filtered["domain"].str.contains(keyword, case=False, na=False, regex=False)
        )
        filtered = filtered[mask]

    # st.date_input returns a single date or a (start, end) tuple while the
    # user is mid-selection — normalise both cases to a [start, end] range.
    if isinstance(date_range, (tuple, list)):
        start_date = date_range[0]
        end_date = date_range[-1]
    else:
        start_date = end_date = date_range
    pub = filtered["published_at"].dt.date
    filtered = filtered[(pub >= start_date) & (pub <= end_date)]

    filtered = filtered[filtered["score"] >= threshold]

    st.sidebar.markdown(f"**{len(filtered)}** of {len(df)} stories match")

    if filtered.empty:
        st.warning("No stories match the current filters. Try loosening them.")
        st.stop()

    kpi_row(filtered)
    st.divider()

    left, right = st.columns([3, 2])
    with left:
        chart_top_stories(filtered)
    with right:
        chart_score_distribution(filtered)
        chart_top_authors(filtered)

    c1, c2 = st.columns(2)
    with c1:
        chart_score_vs_comments(filtered)
    with c2:
        chart_domains(filtered)

    st.divider()
    st.subheader("📋 Story details")
    story_table(filtered)


if __name__ == "__main__":
    main()
else:
    # `streamlit run app/dashboard.py` imports the module — execute the app.
    main()
