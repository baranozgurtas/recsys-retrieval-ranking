"""
Evaluate the trained ranker against retrieval-only baselines.

Outputs:
    results/final_eval.csv      - all metrics for all systems
    results/ranking_eval.csv    - ranker-specific metrics

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --config config/default.yaml --split val
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from flowrec.data.loader import load_config, load_movies
from flowrec.evaluation.metrics import evaluate_recommendations
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

    logger.info("Loading data")
    train_sessions = pd.read_parquet(processed_dir / "train_sessions.parquet")
    context = pd.read_parquet(processed_dir / f"{split}_context.parquet")
    targets_df = pd.read_parquet(processed_dir / f"{split}_targets.parquet")
    targets = build_target_dict(targets_df)
    movies = load_movies(cfg["data"]["raw_dir"], separator=cfg["data"]["separator"])

    item_features = compute_item_features(train_sessions, movies)
    user_features = compute_user_features(train_sessions)

    rows = []

    # popularity baseline
    logger.info("Evaluating popularity baseline")
    pop = PopularityRetriever()
    pop.fit(train_sessions)
    pop_recs = pop.retrieve(context, top_k=top_k_ranking)
    metrics = evaluate_recommendations(pop_recs, targets, k_values)
    for metric, value in metrics.items():
        rows.append({"system": "popularity", "metric": metric, "value": round(value, 5)})

    # co-occurrence retriever
    logger.info("Evaluating co-occurrence retriever")
    cooc = CooccurrenceRetriever(context_window=5)
    cooc.fit(train_sessions)
    cooc_recs = cooc.retrieve(context, top_k=top_k_ranking)
    metrics = evaluate_recommendations(cooc_recs, targets, k_values)
    for metric, value in metrics.items():
        rows.append({"system": "cooccurrence", "metric": metric, "value": round(value, 5)})

    # ranker on top of co-occurrence candidates
    model_path = results_dir / "models" / "ranker.lgb"
    if model_path.exists():
        logger.info("Evaluating LightGBM ranker")
        cooc_scores = dict(cooc._cooc)
        candidates = cooc.retrieve(context, top_k=top_k_retrieval)
        feature_df = build_ranking_features(
            candidates, context, targets_df,
            item_features, user_features, cooc_scores,
        )
        ranker = LightGBMRanker()
        ranker.load(str(model_path))
        ranked_recs = ranker.rerank(feature_df, top_k=top_k_ranking)
        metrics = evaluate_recommendations(ranked_recs, targets, k_values)
        for metric, value in metrics.items():
            rows.append({"system": "cooc+ranker", "metric": metric, "value": round(value, 5)})
    else:
        logger.warning("No trained model found at %s. Run scripts/train.py first.", model_path)

    results_df = pd.DataFrame(rows)
    out_path = results_dir / "final_eval.csv"
    results_df.to_csv(out_path, index=False)
    logger.info("Results written to %s", out_path)

    pivot = results_df.pivot(index="system", columns="metric", values="value")
    ordered_cols = sorted(
        pivot.columns,
        key=lambda c: (c.split("@")[0] if "@" in c else c, int(c.split("@")[1]) if "@" in c else 0)
    )
    pivot = pivot[ordered_cols]
    print(f"\n--- Final Evaluation ({split}) ---")
    print(pivot.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    args = parser.parse_args()
    main(args.config, args.split)
