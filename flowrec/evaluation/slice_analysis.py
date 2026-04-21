import logging

import pandas as pd

from flowrec.evaluation.metrics import evaluate_recommendations

logger = logging.getLogger(__name__)


def _session_lengths(context: pd.DataFrame) -> pd.Series:
    return context.groupby("session_id").size().rename("session_length")


def _item_popularity(train_sessions: pd.DataFrame) -> pd.Series:
    return train_sessions.groupby("item_id").size().rename("item_count")


def _user_interaction_counts(train_sessions: pd.DataFrame) -> pd.Series:
    return train_sessions.groupby("user_id").size().rename("user_count")


def slice_by_session_length(
    context: pd.DataFrame,
    targets: pd.DataFrame,
    recommendations: dict[int, list[int]],
    k_values: list[int],
) -> pd.DataFrame:
    """
    Break down evaluation metrics by session length bucket.

    Buckets:
        short  : 1-3 context items
        medium : 4-9 context items
        long   : 10+ context items
    """
    lengths = _session_lengths(context)
    targets_indexed = targets.set_index("session_id")

    def bucket(n):
        if n <= 3:
            return "short"
        if n <= 9:
            return "medium"
        return "long"

    lengths_df = lengths.reset_index()
    lengths_df["bucket"] = lengths_df["session_length"].apply(bucket)
    sid_to_bucket = lengths_df.set_index("session_id")["bucket"].to_dict()

    rows = []
    for bname in ["short", "medium", "long"]:
        sids = [sid for sid, b in sid_to_bucket.items() if b == bname]
        if not sids:
            continue
        slice_targets = {
            sid: {int(targets_indexed.loc[sid, "target_item_id"])}
            for sid in sids
            if sid in targets_indexed.index
        }
        slice_recs = {sid: recommendations.get(sid, []) for sid in slice_targets}
        metrics = evaluate_recommendations(slice_recs, slice_targets, k_values)
        for metric, value in metrics.items():
            rows.append({
                "slice_type": "session_length",
                "slice_value": bname,
                "n_sessions": len(slice_targets),
                "metric": metric,
                "value": round(value, 5),
            })

    return pd.DataFrame(rows)


def slice_by_item_popularity(
    train_sessions: pd.DataFrame,
    targets: pd.DataFrame,
    recommendations: dict[int, list[int]],
    k_values: list[int],
) -> pd.DataFrame:
    """
    Break down evaluation metrics by target item popularity.

    Buckets:
        head : top 20% most popular items
        tail : bottom 80%
    """
    item_counts = _item_popularity(train_sessions)
    threshold = item_counts.quantile(0.8)
    item_bucket = (item_counts >= threshold).map({True: "head", False: "tail"})

    targets_indexed = targets.set_index("session_id")

    rows = []
    for bname in ["head", "tail"]:
        slice_targets = {}
        for sid in targets_indexed.index:
            target_item = int(targets_indexed.loc[sid, "target_item_id"])
            if item_bucket.get(target_item, "tail") == bname:
                slice_targets[sid] = {target_item}

        if not slice_targets:
            continue

        slice_recs = {sid: recommendations.get(sid, []) for sid in slice_targets}
        metrics = evaluate_recommendations(slice_recs, slice_targets, k_values)
        for metric, value in metrics.items():
            rows.append({
                "slice_type": "item_popularity",
                "slice_value": bname,
                "n_sessions": len(slice_targets),
                "metric": metric,
                "value": round(value, 5),
            })

    return pd.DataFrame(rows)


def slice_by_user_density(
    train_sessions: pd.DataFrame,
    targets: pd.DataFrame,
    recommendations: dict[int, list[int]],
    k_values: list[int],
) -> pd.DataFrame:
    """
    Break down evaluation metrics by user interaction density.

    Buckets:
        sparse : bottom 40% by total training interactions
        dense  : top 40%
        (middle 20% excluded for cleaner separation)
    """
    user_counts = _user_interaction_counts(train_sessions)
    low = user_counts.quantile(0.4)
    high = user_counts.quantile(0.6)

    def bucket(n):
        if n <= low:
            return "sparse"
        if n >= high:
            return "dense"
        return "mid"

    user_bucket = user_counts.apply(bucket)
    targets_indexed = targets.set_index("session_id")
    sid_to_user = targets.set_index("session_id")["user_id"].to_dict()

    rows = []
    for bname in ["sparse", "dense"]:
        slice_targets = {}
        for sid in targets_indexed.index:
            uid = sid_to_user.get(sid)
            if user_bucket.get(uid, "mid") == bname:
                slice_targets[sid] = {int(targets_indexed.loc[sid, "target_item_id"])}

        if not slice_targets:
            continue

        slice_recs = {sid: recommendations.get(sid, []) for sid in slice_targets}
        metrics = evaluate_recommendations(slice_recs, slice_targets, k_values)
        for metric, value in metrics.items():
            rows.append({
                "slice_type": "user_density",
                "slice_value": bname,
                "n_sessions": len(slice_targets),
                "metric": metric,
                "value": round(value, 5),
            })

    return pd.DataFrame(rows)


def run_all_slices(
    context: pd.DataFrame,
    targets: pd.DataFrame,
    train_sessions: pd.DataFrame,
    recommendations: dict[int, list[int]],
    k_values: list[int],
) -> pd.DataFrame:
    """Run all slice analyses and return a combined DataFrame."""
    frames = [
        slice_by_session_length(context, targets, recommendations, k_values),
        slice_by_item_popularity(train_sessions, targets, recommendations, k_values),
        slice_by_user_density(train_sessions, targets, recommendations, k_values),
    ]
    result = pd.concat(frames, ignore_index=True)
    logger.info("Slice analysis complete: %d rows", len(result))
    return result
