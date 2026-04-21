import logging

import pandas as pd

logger = logging.getLogger(__name__)


def compute_user_features(sessions: pd.DataFrame) -> pd.DataFrame:
    """
    Compute user-level features from training sessions.

    Features:
        user_total_interactions  : total items interacted with
        user_total_sessions      : total number of sessions
        user_mean_session_length : mean items per session
        user_is_dense            : 1 if user is in top 20% by interaction count

    Returns
    -------
    DataFrame indexed by user_id
    """
    session_lengths = sessions.groupby(["user_id", "session_id"]).size().reset_index(
        name="session_length"
    )

    user_stats = session_lengths.groupby("user_id").agg(
        user_total_interactions=("session_length", "sum"),
        user_total_sessions=("session_id", "count"),
        user_mean_session_length=("session_length", "mean"),
    ).reset_index()

    threshold = user_stats["user_total_interactions"].quantile(0.8)
    user_stats["user_is_dense"] = (
        user_stats["user_total_interactions"] >= threshold
    ).astype(int)

    user_stats = user_stats.set_index("user_id")

    logger.info(
        "Computed user features: %d users, %d features",
        len(user_stats),
        len(user_stats.columns),
    )
    return user_stats
