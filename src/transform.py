"""Transform stage: clean raw story JSON into a structured, analytics-ready frame."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import project_test.config as config  # noqa: E402

# Columns we keep, mapped from raw HN field names to friendly names.
FIELD_MAP = {
    "id": "id",
    "title": "title",
    "by": "author",
    "score": "score",
    "descendants": "num_comments",
    "time": "time",
    "url": "url",
    "type": "type",
}


def _extract_domain(url: str) -> str:
    """Return the bare domain (without leading 'www.') for a URL, or 'self.HN'."""
    if not url:
        return "self.HN"  # Ask HN / Show HN text posts have no external URL
    netloc = urlparse(url).netloc
    return netloc[4:] if netloc.startswith("www.") else netloc


def transform(raw_stories: list[dict]) -> pd.DataFrame:
    """Clean raw story dicts and return a typed, enriched DataFrame."""
    if not raw_stories:
        raise ValueError("No stories to transform — extract stage returned nothing.")

    df = pd.DataFrame(raw_stories)

    # Keep only the fields we care about; create any missing as empty columns.
    for raw_col in FIELD_MAP:
        if raw_col not in df.columns:
            df[raw_col] = pd.NA
    df = df[list(FIELD_MAP.keys())].rename(columns=FIELD_MAP)

    # --- Handle missing values --------------------------------------------
    df = df.dropna(subset=["id", "title"])           # rows without these are useless
    df["url"] = df["url"].fillna("")                 # Ask/Show HN have no URL
    df["author"] = df["author"].fillna("unknown")
    df["score"] = df["score"].fillna(0).astype(int)
    df["num_comments"] = df["num_comments"].fillna(0).astype(int)
    df["type"] = df["type"].fillna("story")
    df["id"] = df["id"].astype(int)

    # --- Timestamps: unix -> human-readable -------------------------------
    df["published_at"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["published_date"] = df["published_at"].dt.strftime("%Y-%m-%d %H:%M UTC")
    df["published_hour"] = df["published_at"].dt.hour
    df = df.drop(columns=["time"])

    # --- Derived fields ----------------------------------------------------
    df["domain"] = df["url"].apply(_extract_domain)
    df["engagement"] = df["score"] + df["num_comments"]
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    df["ingestion_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Tidy column order.
    column_order = [
        "rank", "id", "title", "author", "score", "num_comments", "engagement",
        "domain", "url", "type", "published_at", "published_date",
        "published_hour", "ingestion_date",
    ]
    return df[column_order]


if __name__ == "__main__":
    from project_test.src.extract import extract

    frame = transform(extract(save_raw=False))
    print(frame.head())
    print(frame.dtypes)
