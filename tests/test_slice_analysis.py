import pandas as pd
import pytest

from flowrec.evaluation.slice_analysis import (
    slice_by_item_popularity,
    slice_by_session_length,
    slice_by_user_density,
    run_all_slices,
)
from flowrec.evaluation.error_analysis import (
    build_error_report,
    coverage_stats,
    find_missed_targets,
    score_distribution,
)


def _make_context():
    rows = []
    # session 0: length 2 (short)
    for i in range(2):
        rows.append({"session_id": 0, "user_id": 1, "item_id": 10 + i, "position": i})
    # session 1: length 5 (medium)
    for i in range(5):
        rows.append({"session_id": 1, "user_id": 2, "item_id": 20 + i, "position": i})
    # session 2: length 12 (long)
    for i in range(12):
        rows.append({"session_id": 2, "user_id": 3, "item_id": 30 + i, "position": i})
    return pd.DataFrame(rows)


def _make_targets():
    return pd.DataFrame([
        {"session_id": 0, "user_id": 1, "target_item_id": 99},
        {"session_id": 1, "user_id": 2, "target_item_id": 99},
        {"session_id": 2, "user_id": 3, "target_item_id": 99},
    ])


def _make_train_sessions():
    rows = []
    # item 99 is very popular
    for u in range(1, 20):
        rows.append({"session_id": u, "user_id": u, "item_id": 99, "position": 0})
    # other items appear once
    for i in range(10, 50):
        rows.append({"session_id": 100 + i, "user_id": 5, "item_id": i, "position": 0})
    return pd.DataFrame(rows)


def _perfect_recs():
    return {0: [99, 1, 2], 1: [99, 3, 4], 2: [99, 5, 6]}


def _empty_recs():
    return {0: [1, 2, 3], 1: [4, 5, 6], 2: [7, 8, 9]}


# --- slice_by_session_length ---

def test_session_length_buckets_present():
    df = slice_by_session_length(_make_context(), _make_targets(), _perfect_recs(), [3])
    buckets = set(df["slice_value"].unique())
    assert {"short", "medium", "long"}.issubset(buckets)


def test_session_length_perfect_recall():
    df = slice_by_session_length(_make_context(), _make_targets(), _perfect_recs(), [3])
    recalls = df[df["metric"] == "recall@3"]["value"]
    assert (recalls == 1.0).all()


def test_session_length_zero_recall():
    df = slice_by_session_length(_make_context(), _make_targets(), _empty_recs(), [3])
    recalls = df[df["metric"] == "recall@3"]["value"]
    assert (recalls == 0.0).all()


# --- slice_by_item_popularity ---

def test_item_popularity_buckets():
    df = slice_by_item_popularity(
        _make_train_sessions(), _make_targets(), _perfect_recs(), [3]
    )
    assert "head" in df["slice_value"].values


def test_item_popularity_head_recall():
    df = slice_by_item_popularity(
        _make_train_sessions(), _make_targets(), _perfect_recs(), [3]
    )
    head_recall = df[(df["slice_value"] == "head") & (df["metric"] == "recall@3")]["value"]
    assert len(head_recall) > 0
    assert (head_recall == 1.0).all()


# --- slice_by_user_density ---

def test_user_density_returns_dataframe():
    train = pd.DataFrame([
        {"session_id": i, "user_id": u, "item_id": 10, "position": 0}
        for i, u in enumerate([1] * 20 + [2] * 2 + [3] * 2)
    ])
    targets = pd.DataFrame([
        {"session_id": 100, "user_id": 1, "target_item_id": 99},
        {"session_id": 101, "user_id": 2, "target_item_id": 99},
    ])
    recs = {100: [99], 101: [99]}
    df = slice_by_user_density(train, targets, recs, [1])
    assert isinstance(df, pd.DataFrame)
    assert "slice_value" in df.columns


# --- run_all_slices ---

def test_run_all_slices_combines_types():
    df = run_all_slices(
        _make_context(), _make_targets(), _make_train_sessions(),
        _perfect_recs(), [3]
    )
    assert set(df["slice_type"].unique()) == {
        "session_length", "item_popularity", "user_density"
    }


# --- error_analysis ---

def test_score_distribution_hit():
    targets = pd.DataFrame([{"session_id": 0, "user_id": 1, "target_item_id": 5}])
    recs = {0: [1, 2, 5, 3]}
    df = score_distribution(recs, targets)
    assert df.loc[0, "target_rank"] == 3


def test_score_distribution_miss():
    targets = pd.DataFrame([{"session_id": 0, "user_id": 1, "target_item_id": 99}])
    recs = {0: [1, 2, 3]}
    df = score_distribution(recs, targets)
    assert df.loc[0, "target_rank"] == -1


def test_find_missed_targets():
    targets = pd.DataFrame([
        {"session_id": 0, "user_id": 1, "target_item_id": 5},
        {"session_id": 1, "user_id": 2, "target_item_id": 9},
    ])
    recs = {0: [5, 2, 3], 1: [1, 2, 3]}
    missed = find_missed_targets(recs, targets, top_k=3)
    assert len(missed) == 1
    assert missed.iloc[0]["target_item_id"] == 9


def test_coverage_stats():
    all_items = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    recs = {0: [1, 2, 3], 1: [3, 4, 5]}
    stats = coverage_stats(recs, all_items, top_k=3)
    assert stats["items_recommended"] == 5
    assert stats["coverage"] == pytest.approx(0.5)


def test_build_error_report_columns():
    context = _make_context()
    targets = _make_targets()
    train = _make_train_sessions()
    recs = _perfect_recs()
    report = build_error_report(recs, targets, train, context, top_k=3)
    for col in ["session_id", "target_rank", "session_length", "hit_at_k"]:
        assert col in report.columns


def test_build_error_report_hit_at_k():
    context = _make_context()
    targets = _make_targets()
    train = _make_train_sessions()
    recs = _perfect_recs()
    report = build_error_report(recs, targets, train, context, top_k=3)
    assert report["hit_at_k"].all()
