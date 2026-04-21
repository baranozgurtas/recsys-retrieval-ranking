"""
Run slice-based evaluation for all trained systems.

Slices:
    - session length  : short / medium / long
    - item popularity : head / tail
    - user density    : sparse / dense

Outputs:
    results/slice_eval_<system>.csv   - per-system slice metrics
    results/slice_eval_combined.csv   - all systems combined

Usage:
    python scripts/slice_evaluate.py
    python scripts/slice_evaluate.py --config config/default.yaml --split val
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from flowrec.data.loader import load_config, load_movies
from flowrec.evaluation.slice_analysis import run_all_slices
from flowrec.features.item_features import compute_item_features
from flowrec.features.user_features import compute_user_features
from flowrec.ranking.feature_builder import build_ranking_features
from flowrec.ranking.ranker import LightGBMRanker
from flowrec.retrieval.cooccurrence import CooccurrenceRetriever
from flowrec.retrieval.popularity import PopularityRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


def build_target_dict(targets: pd.DataFrame) -> dict[int, set[int]]:
    return {
        int(row.session_id): {int(row.target_item_id)}
        for row in targets.itertuples()
    }


def main(config_path: str, split: str) -> None:
    cfg = load_config(config_path)
    processed_dir = Path(cfg["data"]["processed_dir"])
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    top_k_retrieval = cfg["retrieval"]["top_k"]
    top_k_ranking = cfg["ranking"]["top_k"]
    k_values = cfg["evaluation"]["k_values"]

    logger.info("Loading data for split: %s", split)
    train_sessions = pd.read_parquet(processed_dir / "train_sessions.parquet")
    context = pd.read_parquet(processed_dir / f"{split}_context.parquet")
    targets_df = pd.read_parquet(processed_dir / f"{split}_targets.parquet")
    movies = load_movies(cfg["data"]["raw_dir"], separator=cfg["data"]["separator"])

    item_features = compute_item_features(train_sessions, movies)
    user_features = compute_user_features(train_sessions)

    all_frames = []

    # popularity baseline
    logger.info("Slice analysis: popularity")
    pop = PopularityRetriever()
    pop.fit(train_sessions)
    pop_recs = pop.retrieve(context, top_k=top_k_ranking)
    df_pop = run_all_slices(context, targets_df, train_sessions, pop_recs, k_values)
    df_pop["system"] = "popularity"
    all_frames.append(df_pop)

    # co-occurrence
    logger.info("Slice analysis: cooccurrence")
    cooc = CooccurrenceRetriever(context_window=5)
    cooc.fit(train_sessions)
    cooc_recs = cooc.retrieve(context, top_k=top_k_ranking)
    df_cooc = run_all_slices(context, targets_df, train_sessions, cooc_recs, k_values)
    df_cooc["system"] = "cooccurrence"
    all_frames.append(df_cooc)

    # ranker
    model_path = results_dir / "models" / "ranker.lgb"
    if model_path.exists():
        logger.info("Slice analysis: cooc+ranker")
        cooc_scores = dict(cooc._cooc)
        candidates = cooc.retrieve(context, top_k=top_k_retrieval)
        feature_df = build_ranking_features(
            candidates, context, targets_df,
            item_features, user_features, cooc_scores,
        )
        ranker = LightGBMRanker()
        ranker.load(str(model_path))
        ranked_recs = ranker.rerank(feature_df, top_k=top_k_ranking)
        df_ranker = run_all_slices(context, targets_df, train_sessions, ranked_recs, k_values)
        df_ranker["system"] = "cooc+ranker"
        all_frames.append(df_ranker)
    else:
        logger.warning("No trained model found. Run scripts/train.py first.")

    combined = pd.concat(all_frames, ignore_index=True)
    out_path = results_dir / "slice_eval_combined.csv"
    combined.to_csv(out_path, index=False)
    logger.info("Saved %s", out_path)

    for system, grp in combined.groupby("system"):
        path = results_dir / f"slice_eval_{system.replace('+', '_')}.csv"
        grp.to_csv(path, index=False)

    # print summary
    print(f"\n--- Slice Evaluation ({split}) ---")
    for slice_type in combined["slice_type"].unique():
        print(f"\n  [{slice_type}]")
        subset = combined[
            (combined["slice_type"] == slice_type) &
            (combined["metric"].isin([f"recall@{k_values[-1]}", "ndcg@10", "mrr"]))
        ]
        if subset.empty:
            continue
        pivot = subset.pivot_table(
            index=["slice_value", "n_sessions"],
            columns=["system", "metric"],
            values="value",
        )
        print(pivot.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    args = parser.parse_args()
    main(args.config, args.split)
