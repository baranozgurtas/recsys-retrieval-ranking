"""
Generate error analysis reports for the trained ranker.

Outputs:
    results/error_report_<system>.csv     - per-session rank of target item
    results/coverage_stats.csv            - catalog coverage per system
    results/missed_targets_<system>.csv   - sessions where target was not in top_k

Usage:
    python scripts/error_analysis.py
    python scripts/error_analysis.py --config config/default.yaml --split val
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from flowrec.data.loader import load_config, load_movies
from flowrec.evaluation.error_analysis import (
    build_error_report,
    coverage_stats,
    find_missed_targets,
)
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


def main(config_path: str, split: str) -> None:
    cfg = load_config(config_path)
    processed_dir = Path(cfg["data"]["processed_dir"])
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    top_k_retrieval = cfg["retrieval"]["top_k"]
    top_k_ranking = cfg["ranking"]["top_k"]

    logger.info("Loading data")
    train_sessions = pd.read_parquet(processed_dir / "train_sessions.parquet")
    context = pd.read_parquet(processed_dir / f"{split}_context.parquet")
    targets_df = pd.read_parquet(processed_dir / f"{split}_targets.parquet")
    movies = load_movies(cfg["data"]["raw_dir"], separator=cfg["data"]["separator"])

    item_features = compute_item_features(train_sessions, movies)
    user_features = compute_user_features(train_sessions)
    all_items = set(train_sessions["item_id"].unique())

    systems: dict[str, dict] = {}

    pop = PopularityRetriever()
    pop.fit(train_sessions)
    systems["popularity"] = {"recs": pop.retrieve(context, top_k=top_k_ranking)}

    cooc = CooccurrenceRetriever(context_window=5)
    cooc.fit(train_sessions)
    systems["cooccurrence"] = {"recs": cooc.retrieve(context, top_k=top_k_ranking)}

    model_path = results_dir / "models" / "ranker.lgb"
    if model_path.exists():
        cooc_scores = dict(cooc._cooc)
        candidates = cooc.retrieve(context, top_k=top_k_retrieval)
        feature_df = build_ranking_features(
            candidates, context, targets_df,
            item_features, user_features, cooc_scores,
        )
        ranker = LightGBMRanker()
        ranker.load(str(model_path))
        systems["cooc+ranker"] = {"recs": ranker.rerank(feature_df, top_k=top_k_ranking)}
    else:
        logger.warning("No trained model found. Run scripts/train.py first.")

    coverage_rows = []
    for system_name, data in systems.items():
        recs = data["recs"]
        safe_name = system_name.replace("+", "_")

        report = build_error_report(recs, targets_df, train_sessions, context, top_k_ranking)
        report.to_csv(results_dir / f"error_report_{safe_name}.csv", index=False)

        missed = find_missed_targets(recs, targets_df, top_k=top_k_ranking)
        missed.to_csv(results_dir / f"missed_targets_{safe_name}.csv", index=False)

        cov = coverage_stats(recs, all_items, top_k=top_k_ranking)
        cov["system"] = system_name
        coverage_rows.append(cov)

        logger.info(
            "%s: missed=%d / %d (%.1f%%), coverage=%.3f",
            system_name,
            len(missed),
            len(targets_df),
            100 * len(missed) / max(len(targets_df), 1),
            cov["coverage"],
        )

    cov_df = pd.DataFrame(coverage_rows)
    cov_df.to_csv(results_dir / "coverage_stats.csv", index=False)
    logger.info("Saved coverage_stats.csv")

    print("\n--- Coverage Summary ---")
    print(cov_df.to_string(index=False))

    print(f"\n--- Error Report Sample: cooc+ranker (top {top_k_ranking}) ---")
    if "cooc+ranker" in systems:
        report = pd.read_csv(results_dir / "error_report_cooc_ranker.csv")
        print(report.describe().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    args = parser.parse_args()
    main(args.config, args.split)
