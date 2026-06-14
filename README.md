# 📰 Hacker News Data Lake & Analytics Dashboard

An end-to-end **Extract → Transform → Load → Visualize** data pipeline. It pulls
the top 30 stories from the official [Hacker News API](https://github.com/HackerNews/API),
cleans and enriches the data, persists it to a local **Parquet** data lake, and
serves an interactive **Streamlit + Plotly** dashboard.

## Architecture

```
HN Firebase API ──► extract.py ──► transform.py ──► load.py ──► data/processed/*.parquet ──► dashboard.py
   (top 30)         (concurrent     (clean, fix       (snappy        (data lake)            (Streamlit + Plotly)
                     fetch)          missing, unix→dt)  Parquet)
```

| Stage | File | Responsibility |
|-------|------|----------------|
| Extract | [src/extract.py](src/extract.py) | Fetch top story IDs + details concurrently |
| Transform | [src/transform.py](src/transform.py) | Clean, handle missing values, unix→datetime, derive `rank`/`domain`/`engagement` |
| Load | [src/load.py](src/load.py) | Write DataFrame to Parquet (snappy) |
| Orchestrate | [src/run_pipeline.py](src/run_pipeline.py) | Run extract → transform → load |
| Visualize | [app/dashboard.py](app/dashboard.py) | Streamlit dashboard with 5 Plotly charts + table |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the pipeline

```bash
python -m src.run_pipeline
```

This writes `data/processed/hackernews_stories.parquet` (30 rows).

## Launch the dashboard

```bash
streamlit run app/dashboard.py
```

Then open http://localhost:8501.

## Dashboard features

- **KPIs:** story count, average score, total comments, top author
- **Top 15 stories** by score (horizontal bar)
- **Score distribution** (histogram)
- **Top authors** (bar — total score, labelled with story count)
- **Score vs. comments** (bubble scatter sized by engagement)
- **Top source domains** (bar)
- **Story table** with clickable links and a score filter in the sidebar

## Notes

- HN "top stories" are live, so each run produces fresh data.
- Re-run the pipeline any time to refresh; the dashboard reads the latest Parquet.
- Configuration (story count, paths, endpoints) lives in [config.py](config.py).
