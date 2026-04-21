import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_sessions(
    ratings: pd.DataFrame,
    gap_seconds: int = 1800,
    min_length: int = 2,
    max_length: Optional[int] = 50,
) -> pd.DataFrame:
    """
    Segment each user's rating history into sessions based on inactivity gaps.

    A new session begins when the gap between consecutive interactions exceeds
    gap_seconds. Sessions shorter than min_length are discarded. Sessions longer
    than max_length are truncated to the most recent max_length items.

    Returns a DataFrame with columns:
        user_id, session_id (globally unique), item_id, timestamp, position
    """
    df = ratings[["user_id", "item_id", "timestamp"]].copy()
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    df["prev_ts"] = df.groupby("user_id")["timestamp"].shift(1)
    df["delta"] = df["timestamp"] - df["prev_ts"]
    df["new_session"] = df["delta"].isna() | (df["delta"] > gap_seconds)
    df["local_session_id"] = df.groupby("user_id")["new_session"].cumsum()

    # filter short sessions
    lengths = df.groupby(["user_id", "local_session_id"]).transform("size")
    df = df[lengths >= min_length].reset_index(drop=True)

    # truncate long sessions: keep last max_length items per session
    if max_length is not None:
        df = df.sort_values(["user_id", "local_session_id", "timestamp"])
        rev_rank = df.groupby(["user_id", "local_session_id"]).cumcount(ascending=False)
        df = df[rev_rank < max_length].reset_index(drop=True)

    df["position"] = df.groupby(["user_id", "local_session_id"]).cumcount()

    uid_sid_pairs = (
        df[["user_id", "local_session_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    uid_sid_pairs["session_id"] = uid_sid_pairs.index

    df = df.merge(uid_sid_pairs, on=["user_id", "local_session_id"])
    df = (
        df[["user_id", "session_id", "item_id", "timestamp", "position"]]
        .sort_values(["session_id", "position"])
        .reset_index(drop=True)
    )

    logger.info(
        "Built %d sessions across %d users (gap=%ds, min=%d, max=%s)",
        df["session_id"].nunique(),
        df["user_id"].nunique(),
        gap_seconds,
        min_length,
        max_length,
    )
    return df


def split_sessions(
    sessions: pd.DataFrame,
    test_users_fraction: float = 0.2,
    val_users_fraction: float = 0.1,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partition sessions into train, validation, and test splits by user.

    All sessions belonging to a given user remain within a single split,
    preventing any cross-split leakage of user history.
    """
    rng = np.random.default_rng(random_seed)
    users = sessions["user_id"].unique()
    rng.shuffle(users)

    n = len(users)
    n_test = int(n * test_users_fraction)
    n_val = int(n * val_users_fraction)

    test_users = set(users[:n_test])
    val_users = set(users[n_test : n_test + n_val])
    train_users = set(users[n_test + n_val :])

    train = sessions[sessions["user_id"].isin(train_users)].reset_index(drop=True)
    val = sessions[sessions["user_id"].isin(val_users)].reset_index(drop=True)
    test = sessions[sessions["user_id"].isin(test_users)].reset_index(drop=True)

    logger.info(
        "Split: train=%d users / %d sessions | val=%d users / %d sessions | test=%d users / %d sessions",
        len(train_users), train["session_id"].nunique(),
        len(val_users), val["session_id"].nunique(),
        len(test_users), test["session_id"].nunique(),
    )
    return train, val, test


def extract_targets(
    sessions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split each session into context (all but last item) and target (last item).

    Returns
    -------
    context : DataFrame matching sessions schema, last item removed per session
    targets : DataFrame with columns [session_id, user_id, target_item_id]
    """
    last_pos = sessions.groupby("session_id")["position"].transform("max")
    is_last = sessions["position"] == last_pos

    targets = (
        sessions.loc[is_last, ["session_id", "user_id", "item_id"]]
        .rename(columns={"item_id": "target_item_id"})
        .reset_index(drop=True)
    )
    context = sessions[~is_last].reset_index(drop=True)

    logger.info(
        "Extracted %d targets from %d sessions",
        len(targets),
        targets["session_id"].nunique(),
    )
    return context, targets
