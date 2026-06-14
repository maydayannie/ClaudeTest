"""Load stage: persist the transformed DataFrame to the Parquet data lake."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.append(str(Path(__file__).resolve().parent.parent))
import project_test.config as config  # noqa: E402


def save_to_parquet(df: pd.DataFrame, path: Path = config.PARQUET_DATASET) -> Path:
    """Write the DataFrame to a date-partitioned Parquet dataset.

    The dataset is Hive-partitioned by ``ingestion_date`` so each run's data
    lands in its own ``ingestion_date=YYYY-MM-DD/`` folder. Re-running on the
    same day replaces that day's partition (``delete_matching``) instead of
    appending duplicate rows, keeping the load idempotent per day.
    """
    path.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_to_dataset(
        table,
        root_path=str(path),
        partition_cols=[config.PARTITION_COL],
        compression="snappy",
        existing_data_behavior="delete_matching",
    )
    print(f"→ Wrote {len(df)} rows to {path} (partitioned by {config.PARTITION_COL})")
    return path


if __name__ == "__main__":
    from project_test.src.extract import extract
    from project_test.src.transform import transform

    save_to_parquet(transform(extract()))
