import logging

import pandas as pd

logger = logging.getLogger(__name__)


def find_missed_targets(
    recommendations: dict[int, list[int]],
    targets: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    """
    Return sessions where the target item was not in the top_k recommendations.

    Useful for understanding systematic failure modes.
    """
    rows = []
    targets_indexed = targets.set_index("session_id")

    for sid in targets_indexed.index:
        target = int(targets_indexed.loc[sid, "target_item_id"])
        recs = recommendations.get(sid, [])
        if target not in recs[:top_k]:
            rank = recs.index(target) + 1 if target in recs else -1
            rows.append({
                "session_id": sid,
                "target_item_id": target,
                "target_rank": rank,
                "retrieved": target in recs,
            })

    return pd.DataFrame(rows)


def score_distribution(
    recommendations: dict[int, list[int]],
    targets: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each session, compute the rank of the target item in the recommendation list.

    Returns a DataFrame with session_id, target_item_id, and target_rank.
    target_rank == -1 means the target was not in the candidate list at all.
    """
    rows = []
    targets_indexed = targets.set_index("session_id")

    for sid in targets_indexed.index:
        target = int(targets_indexed.loc[sid, "target_item_id"])
        recs = recommendations.get(sid, [])
        rank = recs.index(target) + 1 if target in recs else -1
        rows.append({
            "session_id": int(sid),
            "target_item_id": target,
            "target_rank": rank,
        })

    return pd.DataFrame(rows)


def coverage_stats(
    recommendations: dict[int, list[int]],
    all_items: set[int],
    top_k: int,
) -> dict[str, float]:
    """
    Compute catalog coverage: fraction of items that appear in at least one
    recommendation list.
    """
    recommended_items: set[int] = set()
    for recs in recommendations.values():
        recommended_items.update(recs[:top_k])

    coverage = len(recommended_items) / len(all_items) if all_items else 0.0
    return {
        "catalog_size": len(all_items),
        "items_recommended": len(recommended_items),
        "coverage": round(coverage, 5),
    }


def build_error_report(
    recommendations: dict[int, list[int]],
    targets: pd.DataFrame,
    train_sessions: pd.DataFrame,
    context: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    """
    Combine rank distribution with session and item metadata for inspection.

    Returns a DataFrame useful for identifying systematic failure patterns.
    """
    rank_df = score_distribution(recommendations, targets)

    session_lengths = (
        context.groupby("session_id").size().reset_index(name="session_length")
    )
    rank_df = rank_df.merge(session_lengths, on="session_id", how="left")

    item_counts = train_sessions.groupby("item_id").size().rename("item_train_count")
    rank_df = rank_df.join(item_counts, on="target_item_id", how="left")

    item_popularity_threshold = item_counts.quantile(0.8)
    rank_df["item_is_head"] = (rank_df["item_train_count"] >= item_popularity_threshold).astype(int)
    rank_df["hit_at_k"] = (rank_df["target_rank"] > 0) & (rank_df["target_rank"] <= top_k)

    logger.info(
        "Error report: %d sessions, hit@%d=%.3f",
        len(rank_df),
        top_k,
        rank_df["hit_at_k"].mean(),
    )
    return rank_df
