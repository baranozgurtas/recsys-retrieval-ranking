import pandas as pd
import pytest

from flowrec.data.session_builder import (
    build_sessions,
    extract_targets,
    split_sessions,
)


def _make_ratings(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])


def test_session_boundary_by_gap():
    ratings = _make_ratings([
        (1, 10, 5.0, 0),
        (1, 11, 5.0, 600),
        (1, 12, 5.0, 5000),  # > 1800s gap from previous -> new session
        (1, 13, 5.0, 5600),
    ])
    sessions = build_sessions(ratings, gap_seconds=1800, min_length=2, max_length=None)
    assert sessions["session_id"].nunique() == 2


def test_min_length_filter():
    ratings = _make_ratings([
        (1, 10, 5.0, 0),       # single-item session, should be dropped
        (1, 11, 5.0, 10000),
        (1, 12, 5.0, 10600),
    ])
    sessions = build_sessions(ratings, gap_seconds=1800, min_length=2, max_length=None)
    assert sessions["session_id"].nunique() == 1


def test_max_length_truncation():
    timestamps = list(range(0, 10 * 100, 100))
    rows = [(1, i + 10, 5.0, t) for i, t in enumerate(timestamps)]
    ratings = _make_ratings(rows)
    sessions = build_sessions(ratings, gap_seconds=1800, min_length=2, max_length=5)
    session_len = sessions.groupby("session_id").size().iloc[0]
    assert session_len == 5


def test_position_is_zero_indexed_and_contiguous():
    ratings = _make_ratings([
        (1, 10, 5.0, 0),
        (1, 11, 5.0, 300),
        (1, 12, 5.0, 600),
    ])
    sessions = build_sessions(ratings, gap_seconds=1800, min_length=2, max_length=None)
    positions = sessions[sessions["session_id"] == sessions["session_id"].iloc[0]]["position"].tolist()
    assert positions == list(range(len(positions)))


def test_split_no_user_overlap():
    ratings = _make_ratings(
        [(u, i, 5.0, t)
         for u in range(1, 101)
         for i, t in enumerate(range(0, 5 * 300, 300))]
    )
    sessions = build_sessions(ratings, gap_seconds=1800, min_length=2, max_length=None)
    train, val, test = split_sessions(sessions, test_users_fraction=0.2, val_users_fraction=0.1, random_seed=0)

    train_users = set(train["user_id"])
    val_users = set(val["user_id"])
    test_users = set(test["user_id"])

    assert train_users.isdisjoint(val_users)
    assert train_users.isdisjoint(test_users)
    assert val_users.isdisjoint(test_users)


def test_extract_targets_last_item():
    ratings = _make_ratings([
        (1, 10, 5.0, 0),
        (1, 11, 5.0, 300),
        (1, 12, 5.0, 600),
    ])
    sessions = build_sessions(ratings, gap_seconds=1800, min_length=2, max_length=None)
    context, targets = extract_targets(sessions)

    assert targets["target_item_id"].iloc[0] == 12
    assert 12 not in context["item_id"].values


def test_extract_targets_context_length():
    ratings = _make_ratings([
        (1, 10, 5.0, 0),
        (1, 11, 5.0, 300),
        (1, 12, 5.0, 600),
        (1, 13, 5.0, 900),
    ])
    sessions = build_sessions(ratings, gap_seconds=1800, min_length=2, max_length=None)
    context, targets = extract_targets(sessions)

    sid = targets["session_id"].iloc[0]
    assert len(context[context["session_id"] == sid]) == 3
