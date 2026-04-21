import logging

import pandas as pd

logger = logging.getLogger(__name__)


def compute_item_features(
    sessions: pd.DataFrame,
    movies: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute item-level features from training sessions and movie metadata.

    Features:
        item_global_count     : total interactions in training data
        item_global_rank      : popularity rank (1 = most popular)
        item_unique_users     : number of distinct users who interacted
        item_unique_sessions  : number of distinct sessions
        item_is_head          : 1 if item is in top 20% by count, else 0
        genre_*               : binary genre indicators

    Parameters
    ----------
    sessions : training sessions DataFrame
    movies   : movies metadata with genre_list column

    Returns
    -------
    DataFrame indexed by item_id
    """
    counts = sessions.groupby("item_id").agg(
        item_global_count=("session_id", "count"),
        item_unique_users=("user_id", "nunique"),
        item_unique_sessions=("session_id", "nunique"),
    ).reset_index()

    counts["item_global_rank"] = counts["item_global_count"].rank(
        ascending=False, method="min"
    ).astype(int)

    threshold = counts["item_global_count"].quantile(0.8)
    counts["item_is_head"] = (counts["item_global_count"] >= threshold).astype(int)

    # work on a copy to avoid mutating the caller's dataframe
    movies_copy = movies[["item_id", "genre_list"]].copy()

    all_genres = sorted(
        {g for genres in movies_copy["genre_list"] for g in genres if g != "(no genres listed)"}
    )
    for genre in all_genres:
        safe = genre.lower().replace("-", "_").replace(" ", "_")
        movies_copy[f"genre_{safe}"] = movies_copy["genre_list"].apply(
            lambda gs: int(genre in gs)
        )

    genre_cols = [c for c in movies_copy.columns if c.startswith("genre_")]
    item_features = counts.merge(
        movies_copy[["item_id"] + genre_cols], on="item_id", how="left"
    )
    item_features = item_features.set_index("item_id")

    logger.info(
        "Computed item features: %d items, %d features",
        len(item_features),
        len(item_features.columns),
    )
    return item_features
