"""Central configuration for the Hacker News Data Lake & Analytics pipeline."""

from pathlib import Path

# --- Hacker News API endpoints ---------------------------------------------
# Official Firebase API: https://github.com/HackerNews/API
TOP_STORIES_URL = "https://hacker-news.firebaseio.com/v0/topstories.json"
ITEM_URL = "https://hacker-news.firebaseio.com/v0/item/{id}.json"

# Number of top stories to ingest each run.
NUM_STORIES = 30

# Network behaviour.
REQUEST_TIMEOUT = 10  # seconds per HTTP request
MAX_RETRIES = 3
MAX_WORKERS = 16  # concurrent threads for fetching story details

# --- Filesystem layout (local "data lake") ---------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_SNAPSHOT_FILE = RAW_DIR / "topstories_raw.json"

# Parquet "data lake" is a Hive-partitioned dataset directory rather than a
# single file: data/processed/hackernews_stories/ingestion_date=YYYY-MM-DD/*.parquet
# Partitioning by date keeps each run's data in its own folder so the dataset
# stays small/queryable and old days can be pruned independently.
PARQUET_DATASET = PROCESSED_DIR / "hackernews_stories"
PARTITION_COL = "ingestion_date"
