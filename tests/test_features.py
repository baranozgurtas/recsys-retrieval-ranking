import pandas as pd
import pytest

from flowrec.features.item_features import compute_item_features
from flowrec.features.user_features import compute_user_features
from flowrec.ranking.feature_builder import build_ranking_features, build_session_context_features


def _make_sessions():
    return pd.DataFrame([
        {"user_id": 1, "session_id": 0, "item_id": 10, "timestamp": 0,   "position": 0},
        {"user_id": 1, "session_id": 0, "item_id": 11, "timestamp": 300, "position": 1},
        {"user_id": 1, "session_id": 0, "item_id": 12, "timestamp": 600, "position": 2},
        {"user_id": 2, "session_id": 1, "item_id": 10, "timestamp": 0,   "position": 0},
        {"user_id": 2, "session_id": 1, "item_id": 13, "timestamp": 300, "position": 1},
    ])


def _make_movies():
    return pd.DataFrame([
        {"item_id": 10, "title": "A", "genres": "Action|Comedy", "genre_list": ["Action", "Comedy"]},
        {"item_id": 11, "title": "B", "genres": "Action",        "genre_list": ["Action"]},
        {"item_id": 12, "title": "C", "genres": "Drama",         "genre_list": ["Drama"]},
        {"item_id": 13, "title": "D", "genres": "Comedy",        "genre_list": ["Comedy"]},
    ])


def test_item_features_index():
    sessions = _make_sessions()
    movies = _make_movies()
    feats = compute_item_features(sessions, movies)
    assert feats.index.name == "item_id"
    assert set(feats.index) == {10, 11, 12, 13}


def test_item_features_counts():
    sessions = _make_sessions()
    movies = _make_movies()
    feats = compute_item_features(sessions, movies)
    assert feats.loc[10, "item_global_count"] == 2
    assert feats.loc[11, "item_global_count"] == 1


def test_item_features_genres():
    sessions = _make_sessions()
    movies = _make_movies()
    feats = compute_item_features(sessions, movies)
    assert "genre_action" in feats.columns
    assert feats.loc[10, "genre_action"] == 1
    assert feats.loc[12, "genre_action"] == 0


def test_item_is_head_flag():
    sessions = _make_sessions()
    movies = _make_movies()
    feats = compute_item_features(sessions, movies)
    assert feats["item_is_head"].isin([0, 1]).all()


def test_user_features_index():
    sessions = _make_sessions()
    feats = compute_user_features(sessions)
    assert feats.index.name == "user_id"
    assert set(feats.index) == {1, 2}


def test_user_features_session_length():
    sessions = _make_sessions()
    feats = compute_user_features(sessions)
    assert feats.loc[1, "user_mean_session_length"] == pytest.approx(3.0)
    assert feats.loc[2, "user_mean_session_length"] == pytest.approx(2.0)


def test_session_context_features_last_item():
    context = pd.DataFrame([
        {"session_id": 0, "item_id": 10, "position": 0},
        {"session_id": 0, "item_id": 11, "position": 1},
        {"session_id": 0, "item_id": 12, "position": 2},
    ])
    feats = build_session_context_features(context)
    row = feats[feats["session_id"] == 0].iloc[0]
    assert row["last_item_id"] == 12
    assert row["second_last_item_id"] == 11
    assert row["session_length"] == 3


def test_build_ranking_features_shape():
    sessions = _make_sessions()
    movies = _make_movies()
    item_features = compute_item_features(sessions, movies)
    user_features = compute_user_features(sessions)

    context = sessions[sessions["position"] < sessions.groupby("session_id")["position"].transform("max")]
    targets = pd.DataFrame([
        {"session_id": 0, "user_id": 1, "target_item_id": 12},
        {"session_id": 1, "user_id": 2, "target_item_id": 13},
    ])
    candidates = {0: [12, 11, 10], 1: [13, 10, 11]}

    df = build_ranking_features(candidates, context, targets, item_features, user_features)
    assert len(df) == 6
    assert "label" in df.columns
    assert df["label"].sum() == 2


def test_build_ranking_features_labels():
    sessions = _make_sessions()
    movies = _make_movies()
    item_features = compute_item_features(sessions, movies)
    user_features = compute_user_features(sessions)

    context = sessions[sessions["position"] < sessions.groupby("session_id")["position"].transform("max")]
    targets = pd.DataFrame([
        {"session_id": 0, "user_id": 1, "target_item_id": 12},
    ])
    candidates = {0: [12, 11]}

    df = build_ranking_features(candidates, context, targets, item_features, user_features)
    label_for_target = df[df["candidate_item_id"] == 12]["label"].iloc[0]
    label_for_other = df[df["candidate_item_id"] == 11]["label"].iloc[0]
    assert label_for_target == 1
    assert label_for_other == 0
