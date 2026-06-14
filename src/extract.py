"""Extract stage: fetch top story IDs and their detailed JSON from the HN API."""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# Allow running both as a module (`python -m src.run_pipeline`) and directly.
sys.path.append(str(Path(__file__).resolve().parent.parent))
import project_test.config as config  # noqa: E402

# HTTP status codes that indicate an unrecoverable authentication failure.
# These are intentionally NOT in the retry `status_forcelist` — retrying an
# auth error is pointless, so we surface them immediately as a fatal error.
#   401 Unauthorized · 403 Forbidden · 407 Proxy Auth Required · 511 Network Auth Required
AUTH_ERROR_CODES = (401, 403, 407, 511)


class HackerNewsAuthError(Exception):
    """Raised when the HN API rejects a request with an auth error."""


def _get(url: str, session: requests.Session) -> requests.Response:
    """GET a URL, translating auth failures into :class:`HackerNewsAuthError`.

    Transient errors (429, 5xx, connection drops) are left to the session's
    retry policy; only genuine auth failures are converted into a fatal error.
    """
    resp = session.get(url, timeout=config.REQUEST_TIMEOUT)
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        if resp.status_code in AUTH_ERROR_CODES:
            raise HackerNewsAuthError(
                f"Hacker News API returned {resp.status_code} "
                f"({resp.reason}) for {url}"
            ) from exc
        raise
    return resp


def _build_session() -> requests.Session:
    """Create a requests Session with simple retry handling."""
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    retry = Retry(
        total=config.MAX_RETRIES,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch_top_story_ids(limit: int = config.NUM_STORIES,
                        session: requests.Session | None = None) -> list[int]:
    """Return the first ``limit`` top-story IDs from the HN API."""
    sess = session or _build_session()
    resp = _get(config.TOP_STORIES_URL, sess)
    ids = resp.json() or []
    return ids[:limit]


def fetch_story(story_id: int,
                session: requests.Session | None = None) -> dict | None:
    """Fetch the detailed JSON for a single story. Returns None if unavailable."""
    sess = session or _build_session()
    url = config.ITEM_URL.format(id=story_id)
    resp = _get(url, sess)
    return resp.json()  # may be None for deleted/dead items


def fetch_stories(ids: list[int],
                  session: requests.Session | None = None) -> list[dict]:
    """Fetch story details for many IDs concurrently, preserving rank order."""
    sess = session or _build_session()
    results: dict[int, dict] = {}

    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        future_to_id = {
            executor.submit(fetch_story, sid, sess): sid for sid in ids
        }
        for future in as_completed(future_to_id):
            sid = future_to_id[future]
            try:
                item = future.result()
            except requests.RequestException as exc:
                print(f"  ! failed to fetch story {sid}: {exc}", file=sys.stderr)
                continue
            if item:  # skip None / deleted
                results[sid] = item

    # Re-order according to the original (ranked) ID list.
    ordered = [results[sid] for sid in ids if sid in results]
    return ordered


def save_raw_snapshot(stories: list[dict], path: Path = config.RAW_SNAPSHOT_FILE) -> None:
    """Persist the raw API response for traceability / reprocessing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(stories, fh, indent=2)


def extract(limit: int = config.NUM_STORIES, save_raw: bool = True) -> list[dict]:
    """Run the full extract stage and return raw story dicts.

    On a fatal authentication error (401/403/407/511) this prints a clean
    message to stderr and exits the process gracefully (code 1) rather than
    propagating a raw traceback.
    """
    session = _build_session()
    try:
        print(f"→ Fetching top {limit} story IDs ...")
        ids = fetch_top_story_ids(limit, session=session)
        print(f"  got {len(ids)} IDs")

        print("→ Fetching story details (concurrent) ...")
        stories = fetch_stories(ids, session=session)
        print(f"  retrieved {len(stories)} stories")
    except HackerNewsAuthError as exc:
        print(
            "\nERROR: Authentication failed while contacting the Hacker News API.\n"
            f"  {exc}\n"
            "  This is not a transient error, so the pipeline will not retry.\n"
            "  Please verify the API endpoint/credentials in config.py and try again.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

    if save_raw:
        save_raw_snapshot(stories)
        print(f"  raw snapshot → {config.RAW_SNAPSHOT_FILE}")

    return stories


if __name__ == "__main__":
    data = extract()
    print(f"Extracted {len(data)} stories.")
