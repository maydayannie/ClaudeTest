"""Orchestrator: run the full Extract -> Transform -> Load pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import project_test.config as config  # noqa: E402
from project_test.src.extract import extract  # noqa: E402
from project_test.src.load import save_to_parquet  # noqa: E402
from project_test.src.transform import transform  # noqa: E402


def run() -> None:
    print("=" * 60)
    print("Hacker News Data Lake — ETL pipeline")
    print("=" * 60)

    raw = extract(limit=config.NUM_STORIES)
    df = transform(raw)
    print(f"→ Transformed into {len(df)} rows × {len(df.columns)} columns")
    path = save_to_parquet(df)

    top = df.iloc[0]
    print("-" * 60)
    print(f"✓ Pipeline complete — {len(df)} rows written")
    print(f"  Output : {path}")
    print(f"  Top story (#{int(top['rank'])}, score {int(top['score'])}): {top['title']}")
    print("=" * 60)


if __name__ == "__main__":
    run()
