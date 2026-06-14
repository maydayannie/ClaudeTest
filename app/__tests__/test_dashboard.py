"""Comprehensive tests for app/dashboard.py."""
from __future__ import annotations

import datetime
import importlib.util
import sys
import types as _types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Step 1: Mock streamlit before importing dashboard.
# dashboard.py calls st.set_page_config() and @st.cache_data at module level.
# ---------------------------------------------------------------------------
_st = MagicMock()
_st.cache_data.return_value = lambda f: f  # make @st.cache_data a no-op decorator
sys.modules["streamlit"] = _st

# ---------------------------------------------------------------------------
# Step 2: Mock config so the import-time main() call sees no data and
# immediately calls st.stop() before touching sidebar widgets or filters.
# ---------------------------------------------------------------------------
_fake_config = _types.ModuleType("config")
_fake_config.PARQUET_DATASET = Path("/nonexistent/does/not/exist")
_fake_config.PARTITION_COL = "ingestion_date"
sys.modules["config"] = _fake_config

# ---------------------------------------------------------------------------
# Step 3: Make st.stop() raise a BaseException so main() actually halts.
# We catch it during the exec_module call below.
# ---------------------------------------------------------------------------
class _StopExecution(BaseException):
    pass

_st.stop.side_effect = _StopExecution

# ---------------------------------------------------------------------------
# Step 4: Load dashboard via importlib with the module pre-registered in
# sys.modules.  When exec_module raises _StopExecution (from the end-of-file
# `main()` call), all function definitions above that call are already bound
# to the module object.
# ---------------------------------------------------------------------------
_app_dir = str(Path(__file__).resolve().parent.parent)
_spec = importlib.util.spec_from_file_location(
    "dashboard", Path(_app_dir) / "dashboard.py"
)
_module = importlib.util.module_from_spec(_spec)
sys.modules["dashboard"] = _module  # pre-register so exec_module failure doesn't remove it

try:
    _spec.loader.exec_module(_module)
except _StopExecution:
    pass  # expected: import-time main() hit st.stop()

import dashboard  # resolves from the pre-registered sys.modules entry

# Reset st.stop so tests can configure it as needed.
_st.stop.side_effect = None


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

def _make_col_capture():
    """Return (captured_list, side_effect_fn) to inspect column mocks."""
    captured: list[MagicMock] = []

    def _se(spec):
        n = spec if isinstance(spec, int) else len(spec)
        mocks = [MagicMock(name=f"col{i}") for i in range(n)]
        captured.extend(mocks)
        return mocks

    return captured, _se


@pytest.fixture(autouse=True)
def reset_st():
    """Reset the streamlit mock and configure columns() before every test."""
    _st.reset_mock()
    _st.stop.side_effect = None

    def _default_columns(spec):
        n = spec if isinstance(spec, int) else len(spec)
        return [MagicMock() for _ in range(n)]

    _st.columns.side_effect = _default_columns
    yield


def _df(
    n: int = 10,
    scores: list[int] | None = None,
    authors: list[str] | None = None,
    domains: list[str] | None = None,
) -> pd.DataFrame:
    """Build a minimal DataFrame compatible with all dashboard functions."""
    if scores is None:
        scores = list(range(100, 100 + n * 10, 10))
    if authors is None:
        authors = [f"user{i % 3}" for i in range(n)]
    if domains is None:
        domains = [f"site{i % 4}.io" for i in range(n)]

    published_at = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({
        "id": range(1, n + 1),
        "rank": range(1, n + 1),
        "title": [f"Story {'word ' * 15}{i}" for i in range(n)],
        "author": authors,
        "score": scores,
        "num_comments": list(range(5, 5 + n * 3, 3)),
        "engagement": list(range(20, 20 + n * 4, 4)),
        "domain": domains,
        "published_at": published_at,
        "published_date": [d.date() for d in published_at],
        "url": [f"https://example.com/{i}" for i in range(n)],
        "ingestion_date": "2024-01-01",
    })


# ===========================================================================
# load_data
# ===========================================================================

class TestLoadData:
    @patch("dashboard.pd.read_parquet")
    def test_returns_dataframe(self, mock_read):
        mock_read.return_value = _df(n=3)
        result = dashboard.load_data("/some/path")
        assert isinstance(result, pd.DataFrame)
        mock_read.assert_called_once_with("/some/path")

    @patch("dashboard.pd.read_parquet")
    def test_normalizes_categorical_ingestion_date_to_str(self, mock_read):
        df = _df(n=3)
        df["ingestion_date"] = pd.Categorical(["2024-01-01"] * 3)
        mock_read.return_value = df
        result = dashboard.load_data("/path")
        # pandas 2.x may return StringDtype rather than object; check values directly
        assert pd.api.types.is_string_dtype(result["ingestion_date"])
        assert result["ingestion_date"].iloc[0] == "2024-01-01"

    @patch("dashboard.pd.read_parquet")
    def test_no_ingestion_date_column_passes_through_unchanged(self, mock_read):
        df = _df(n=3).drop(columns=["ingestion_date"])
        mock_read.return_value = df
        result = dashboard.load_data("/path")
        assert "ingestion_date" not in result.columns

    @patch("dashboard.pd.read_parquet")
    def test_ingestion_date_already_str_is_preserved(self, mock_read):
        df = _df(n=3)
        df["ingestion_date"] = "2024-03-15"
        mock_read.return_value = df
        result = dashboard.load_data("/path")
        assert result["ingestion_date"].iloc[0] == "2024-03-15"

    def test_raises_on_nonexistent_path(self):
        with pytest.raises(Exception):
            dashboard.load_data("/nonexistent/path/data.parquet")


# ===========================================================================
# kpi_row
# ===========================================================================

class TestKpiRow:
    def test_top_author_is_highest_cumulative_score(self):
        # alice=700, bob=600, carol=100 → alice wins
        df = _df(n=4, authors=["alice", "alice", "bob", "carol"],
                 scores=[400, 300, 600, 100])
        captured, se = _make_col_capture()
        _st.columns.side_effect = se
        dashboard.kpi_row(df)
        captured[3].metric.assert_called_once_with("Top author", "alice")

    def test_story_count_metric(self):
        df = _df(n=7)
        captured, se = _make_col_capture()
        _st.columns.side_effect = se
        dashboard.kpi_row(df)
        captured[0].metric.assert_called_once_with("Stories", 7)

    def test_average_score_metric(self):
        df = _df(n=4, scores=[100, 200, 300, 400])  # mean = 250
        captured, se = _make_col_capture()
        _st.columns.side_effect = se
        dashboard.kpi_row(df)
        captured[1].metric.assert_called_once_with("Avg. score", "250")

    def test_total_comments_metric(self):
        df = _df(n=3)
        expected = int(df["num_comments"].sum())
        captured, se = _make_col_capture()
        _st.columns.side_effect = se
        dashboard.kpi_row(df)
        captured[2].metric.assert_called_once_with("Total comments", f"{expected:,}")

    def test_single_story_no_error(self):
        df = _df(n=1, authors=["solo"], scores=[42])
        captured, se = _make_col_capture()
        _st.columns.side_effect = se
        dashboard.kpi_row(df)
        captured[3].metric.assert_called_once_with("Top author", "solo")

    def test_tie_in_scores_does_not_raise(self):
        df = _df(n=2, authors=["a", "b"], scores=[100, 100])
        dashboard.kpi_row(df)  # must not raise


# ===========================================================================
# chart_top_stories
# ===========================================================================

class TestChartTopStories:
    def _fig(self):
        return _st.plotly_chart.call_args[0][0]

    def test_selects_exactly_15_stories(self):
        dashboard.chart_top_stories(_df(n=20))
        assert len(self._fig().data[0].x) == 15

    def test_uses_all_stories_when_fewer_than_15(self):
        dashboard.chart_top_stories(_df(n=5))
        assert len(self._fig().data[0].x) == 5

    def test_truncates_titles_to_60_chars(self):
        df = _df(n=3)
        df["title"] = ["A" * 100] * 3
        dashboard.chart_top_stories(df)
        for label in self._fig().data[0].y:
            assert len(label) <= 60

    def test_chart_title_mentions_top_15(self):
        dashboard.chart_top_stories(_df(n=3))
        assert "Top 15" in self._fig().layout.title.text

    def test_highest_score_story_included(self):
        df = _df(n=20)
        max_score = df["score"].max()
        dashboard.chart_top_stories(df)
        assert max_score in list(self._fig().data[0].x)

    def test_lowest_score_excluded_when_enough_rows(self):
        df = _df(n=20)
        min_score = df["score"].min()
        dashboard.chart_top_stories(df)
        assert min_score not in list(self._fig().data[0].x)

    def test_scores_sorted_ascending_for_horizontal_bar(self):
        df = _df(n=5)
        dashboard.chart_top_stories(df)
        scores = list(self._fig().data[0].x)
        assert scores == sorted(scores)


# ===========================================================================
# chart_score_distribution
# ===========================================================================

class TestChartScoreDistribution:
    def _fig(self):
        return _st.plotly_chart.call_args[0][0]

    def test_calls_plotly_chart_once(self):
        dashboard.chart_score_distribution(_df(n=10))
        _st.plotly_chart.assert_called_once()

    def test_chart_title(self):
        dashboard.chart_score_distribution(_df(n=5))
        assert "Score Distribution" in self._fig().layout.title.text

    def test_uses_hn_orange_color(self):
        dashboard.chart_score_distribution(_df(n=5))
        assert dashboard.HN_ORANGE in str(self._fig())

    def test_single_row_no_error(self):
        dashboard.chart_score_distribution(_df(n=1))


# ===========================================================================
# chart_top_authors
# ===========================================================================

class TestChartTopAuthors:
    def _fig(self):
        return _st.plotly_chart.call_args[0][0]

    def test_limits_to_10_authors(self):
        df = _df(n=20, authors=[f"user{i}" for i in range(20)])
        dashboard.chart_top_authors(df)
        assert len(self._fig().data[0].y) == 10

    def test_fewer_than_10_authors_shows_all(self):
        df = _df(n=4, authors=["a", "b", "c", "d"])
        dashboard.chart_top_authors(df)
        assert len(self._fig().data[0].y) == 4

    def test_aggregates_total_score_per_author(self):
        df = _df(
            n=4,
            authors=["alice", "alice", "bob", "bob"],
            scores=[300, 200, 100, 50],
        )
        dashboard.chart_top_authors(df)
        fig = self._fig()
        scores_by_author = dict(zip(fig.data[0].y, fig.data[0].x))
        assert scores_by_author["alice"] == 500
        assert scores_by_author["bob"] == 150

    def test_top_scoring_author_appears(self):
        df = _df(n=3, authors=["alice", "bob", "alice"], scores=[999, 1, 999])
        dashboard.chart_top_authors(df)
        assert "alice" in list(self._fig().data[0].y)

    def test_chart_title(self):
        dashboard.chart_top_authors(_df(n=5))
        assert "Top Authors" in self._fig().layout.title.text


# ===========================================================================
# chart_score_vs_comments
# ===========================================================================

class TestChartScoreVsComments:
    def _fig(self):
        return _st.plotly_chart.call_args[0][0]

    def test_calls_plotly_chart_once(self):
        dashboard.chart_score_vs_comments(_df(n=10))
        _st.plotly_chart.assert_called_once()

    def test_chart_title(self):
        dashboard.chart_score_vs_comments(_df(n=5))
        assert "Score vs. Comments" in self._fig().layout.title.text

    def test_produces_scatter_data(self):
        dashboard.chart_score_vs_comments(_df(n=8))
        assert len(self._fig().data) > 0

    def test_single_row_no_error(self):
        dashboard.chart_score_vs_comments(_df(n=1))


# ===========================================================================
# chart_domains
# ===========================================================================

class TestChartDomains:
    def _fig(self):
        return _st.plotly_chart.call_args[0][0]

    def test_limits_to_10_domains(self):
        df = _df(n=15, domains=[f"d{i}.com" for i in range(15)])
        dashboard.chart_domains(df)
        assert len(self._fig().data[0].y) == 10

    def test_fewer_than_10_domains_shows_all(self):
        df = _df(n=6, domains=["a.com", "b.com", "c.com", "a.com", "b.com", "c.com"])
        dashboard.chart_domains(df)
        assert len(self._fig().data[0].y) == 3

    def test_count_values_are_correct(self):
        df = _df(
            n=5,
            domains=["top.com", "top.com", "top.com", "low.com", "low.com"],
        )
        dashboard.chart_domains(df)
        fig = self._fig()
        counts_by_domain = dict(zip(fig.data[0].y, fig.data[0].x))
        assert counts_by_domain["top.com"] == 3
        assert counts_by_domain["low.com"] == 2

    def test_chart_title_mentions_domain(self):
        dashboard.chart_domains(_df(n=5))
        assert "Domain" in self._fig().layout.title.text


# ===========================================================================
# story_table
# ===========================================================================

class TestStoryTable:
    REQUIRED_COLS = [
        "rank", "title", "author", "score", "num_comments",
        "domain", "published_date", "url",
    ]

    def test_selects_exactly_required_columns(self):
        dashboard.story_table(_df(n=5))
        view = _st.dataframe.call_args[0][0]
        assert list(view.columns) == self.REQUIRED_COLS

    def test_excludes_internal_columns(self):
        dashboard.story_table(_df(n=5))
        view = _st.dataframe.call_args[0][0]
        for col in ("engagement", "ingestion_date", "published_at", "id"):
            assert col not in view.columns

    def test_row_count_preserved(self):
        dashboard.story_table(_df(n=12))
        view = _st.dataframe.call_args[0][0]
        assert len(view) == 12

    def test_raises_on_missing_required_column(self):
        df = _df(n=3).drop(columns=["url"])
        with pytest.raises(KeyError):
            dashboard.story_table(df)

    def test_hide_index_kwarg_is_set(self):
        dashboard.story_table(_df(n=3))
        kwargs = _st.dataframe.call_args[1]
        assert kwargs.get("hide_index") is True


# ===========================================================================
# Filter logic
# The three filters live inside main(), so we replicate the same logic here
# and test correctness independently.
# ===========================================================================

def _apply_keyword(df: pd.DataFrame, keyword: str) -> pd.DataFrame:
    mask = (
        df["title"].str.contains(keyword, case=False, na=False, regex=False)
        | df["author"].str.contains(keyword, case=False, na=False, regex=False)
        | df["domain"].str.contains(keyword, case=False, na=False, regex=False)
    )
    return df[mask]


def _apply_date_range(df: pd.DataFrame, date_range) -> pd.DataFrame:
    if isinstance(date_range, (tuple, list)):
        start_date = date_range[0]
        end_date = date_range[-1]
    else:
        start_date = end_date = date_range
    pub = df["published_at"].dt.date
    return df[(pub >= start_date) & (pub <= end_date)]


def _apply_score_threshold(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    return df[df["score"] >= threshold]


class TestKeywordFilter:
    def test_matches_title_substring(self):
        df = _df(n=3)
        df["title"] = ["Python rocks", "Rust is fast", "Go concurrency"]
        result = _apply_keyword(df, "python")
        assert len(result) == 1
        assert result.iloc[0]["title"] == "Python rocks"

    def test_matches_author(self):
        df = _df(n=3, authors=["alice", "bob", "carol"])
        result = _apply_keyword(df, "Bob")
        assert len(result) == 1
        assert result.iloc[0]["author"] == "bob"

    def test_matches_domain(self):
        df = _df(n=3, domains=["github.com", "medium.com", "ycombinator.com"])
        result = _apply_keyword(df, "github")
        assert len(result) == 1

    def test_case_insensitive(self):
        df = _df(n=2)
        df["title"] = ["UPPERCASE TITLE", "lowercase title"]
        assert len(_apply_keyword(df, "uppercase")) == 1
        assert len(_apply_keyword(df, "LOWERCASE")) == 1

    def test_no_match_returns_empty(self):
        df = _df(n=3)
        df["title"] = ["alpha", "beta", "gamma"]
        assert _apply_keyword(df, "zzz").empty

    def test_special_characters_treated_as_literals(self):
        # regex=False means (.*) is searched as a literal string
        df = _df(n=2)
        df["title"] = ["regex (.*) test", "no match here"]
        result = _apply_keyword(df, "(.*)")
        assert len(result) == 1

    def test_matches_across_multiple_columns(self):
        df = _df(n=1, authors=["search_term"], domains=["example.com"])
        df["title"] = ["unrelated title"]
        assert len(_apply_keyword(df, "search_term")) == 1


class TestDateRangeFilter:
    def test_tuple_start_end_filters_correctly(self):
        df = _df(n=5)
        dates = df["published_at"].dt.date.tolist()
        start, end = dates[1], dates[3]
        result = _apply_date_range(df, (start, end))
        assert all((result["published_at"].dt.date >= start) &
                   (result["published_at"].dt.date <= end))

    def test_bounds_are_inclusive(self):
        df = _df(n=5)
        dates = df["published_at"].dt.date.tolist()
        result = _apply_date_range(df, (dates[0], dates[-1]))
        assert len(result) == 5

    def test_single_date_value_matches_that_day_only(self):
        df = _df(n=5)
        target = df["published_at"].dt.date.iloc[0]
        result = _apply_date_range(df, target)
        assert all(result["published_at"].dt.date == target)

    def test_single_element_tuple_uses_last_element_as_end(self):
        # Streamlit returns a 1-tuple mid-selection; dashboard uses date_range[-1].
        df = _df(n=4)
        target = df["published_at"].dt.date.iloc[2]
        result = _apply_date_range(df, (target,))
        assert all(result["published_at"].dt.date <= target)

    def test_non_overlapping_range_returns_empty(self):
        df = _df(n=3)
        future = datetime.date(2099, 1, 1)
        assert _apply_date_range(df, (future, future)).empty


class TestScoreThresholdFilter:
    def test_excludes_stories_below_threshold(self):
        df = _df(n=5, scores=[10, 20, 30, 40, 50])
        result = _apply_score_threshold(df, 30)
        assert all(result["score"] >= 30)

    def test_correct_count_above_threshold(self):
        df = _df(n=5, scores=[10, 20, 30, 40, 50])
        assert len(_apply_score_threshold(df, 30)) == 3  # 30, 40, 50

    def test_minimum_score_includes_all_rows(self):
        df = _df(n=5, scores=[10, 20, 30, 40, 50])
        assert len(_apply_score_threshold(df, 10)) == 5

    def test_threshold_equal_to_max_returns_one_row(self):
        df = _df(n=5, scores=[10, 20, 30, 40, 50])
        assert len(_apply_score_threshold(df, 50)) == 1

    def test_threshold_above_max_returns_empty(self):
        df = _df(n=5, scores=[10, 20, 30, 40, 50])
        assert _apply_score_threshold(df, 999).empty

    def test_threshold_is_inclusive(self):
        df = _df(n=3, scores=[100, 200, 300])
        result = _apply_score_threshold(df, 200)
        assert 200 in result["score"].tolist()
