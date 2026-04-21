import logging

import pandas as pd

logger = logging.getLogger(__name__)


def build_session_context_features(context: pd.DataFrame) -> pd.DataFrame:
    """
    Compute session-level features derived from the context window.

    Features:
        session_length          : number of items in context
        last_item_id            : most recent item in session
        second_last_item_id     : second most recent item (0 if unavailable)
    """
    sorted_ctx = context.sort_values(["session_id", "position"])

    session_length = (
        sorted_ctx.groupby("session_id").size().reset_index(name="session_length")
    )

    last_item = (
        sorted_ctx.groupby("session_id")["item_id"]
        .last()
        .reset_index()
        .rename(columns={"item_id": "last_item_id"})
    )

    def _second_last(grp):
        items = grp.values
        return items[-2] if len(items) >= 2 else 0

    second_last = (
        sorted_ctx.groupby("session_id")["item_id"]
        .apply(_second_last)
        .reset_index()
        .rename(columns={"item_id": "second_last_item_id"})
    )

    features = session_length.merge(last_item, on="session_id", how="left")
    features = features.merge(second_last, on="session_id", how="left")
    features["second_last_item_id"] = features["second_last_item_id"].fillna(0).astype(int)
    return features


def build_ranking_features(
    candidates: dict[int, list[int]],
    context: pd.DataFrame,
    targets: pd.DataFrame,
    item_features: pd.DataFrame,
    user_features: pd.DataFrame,
    cooc_scores: dict[int, dict[int, float]] | None = None,
) -> pd.DataFrame:
    """
    Assemble the training/inference feature matrix for the ranking model.

    Each row represents a (session, candidate_item) pair.

    Parameters
    ----------
    candidates   : {session_id: [candidate item_ids]}
    context      : session context rows
    targets      : DataFrame with [session_id, user_id, target_item_id]
    item_features: DataFrame indexed by item_id
    user_features: DataFrame indexed by user_id
    cooc_scores  : optional {item_id: {neighbor_id: score}} for cooc feature

    Returns
    -------
    DataFrame with one row per (session_id, candidate_item_id), including
    a binary `label` column (1 if candidate == target).
    """
    rows = []
    for session_id, items in candidates.items():
        for item_id in items:
            rows.append({"session_id": int(session_id), "candidate_item_id": int(item_id)})

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    session_ctx = build_session_context_features(context)
    df = df.merge(session_ctx, on="session_id", how="left")

    sid_to_user = targets.set_index("session_id")["user_id"].to_dict()
    df["user_id"] = df["session_id"].map(sid_to_user)

    item_feat_cols = [c for c in item_features.columns if item_features[c].dtype != object]
    df = df.join(item_features[item_feat_cols], on="candidate_item_id", how="left")

    user_feat_cols = user_features.columns.tolist()
    df = df.join(user_features[user_feat_cols], on="user_id", how="left")

    if cooc_scores is not None:
        def get_cooc(row):
            return cooc_scores.get(row["last_item_id"], {}).get(row["candidate_item_id"], 0.0)
        df["cooc_score_last_item"] = df.apply(get_cooc, axis=1)
    else:
        df["cooc_score_last_item"] = 0.0

    sid_to_target = targets.set_index("session_id")["target_item_id"].to_dict()
    df["label"] = (
        df["candidate_item_id"] == df["session_id"].map(sid_to_target)
    ).astype(int)

    df = df.fillna(0)

    logger.info(
        "Built ranking feature matrix: %d rows, %d features, %d positive labels",
        len(df),
        len(df.columns),
        df["label"].sum(),
    )
    return df
