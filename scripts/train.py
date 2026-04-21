"""
Train the LightGBM ranking model on top of co-occurrence candidates.

Outputs:
    results/models/ranker.lgb           - saved model
    results/train_feature_matrix.parquet
    results/val_feature_matrix.parquet

Usage:
    python scripts/train.py
    python scripts/train.py --config config/default.yaml
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from flowrec.data.loader import load_config, load_movies
from flowrec.features.item_features import compute_item_features
from flowrec.features.user_features import compute_user_features
from flowrec.ranking.feature_builder import build_ranking_features
from flowrec.ranking.ranker import LightGBMRanker
from flowrec.retrieval.cooccurrence import CooccurrenceRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    processed_dir = Path(cfg["data"]["processed_dir"])
    results_dir = Path("results")
    models_dir = results_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    top_k = cfg["retrieval"]["top_k"]

    logger.info("Loading data")
    train_sessions = pd.read_parquet(processed_dir / "train_sessions.parquet")
    train_context = pd.read_parquet(processed_dir / "train_context.parquet")
    train_targets = pd.read_parquet(processed_dir / "train_targets.parquet")
    val_context = pd.read_parquet(processed_dir / "val_context.parquet")
    val_targets = pd.read_parquet(processed_dir / "val_targets.parquet")
    movies = load_movies(cfg["data"]["raw_dir"], separator=cfg["data"]["separator"])

    logger.info("Computing features")
    item_features = compute_item_features(train_sessions, movies)
    user_features = compute_user_features(train_sessions)

    logger.info("Fitting co-occurrence retriever")
    retriever = CooccurrenceRetriever(context_window=5)
    retriever.fit(train_sessions)
    cooc_scores = dict(retriever._cooc)

    logger.info("Generating training candidates (top_k=%d)", top_k)
    train_candidates = retriever.retrieve(train_context, top_k=top_k)

    logger.info("Generating validation candidates")
    val_candidates = retriever.retrieve(val_context, top_k=top_k)

    logger.info("Building training feature matrix")
    train_df = build_ranking_features(
        train_candidates, train_context, train_targets,
        item_features, user_features, cooc_scores,
    )

    logger.info("Building validation feature matrix")
    val_df = build_ranking_features(
        val_candidates, val_context, val_targets,
        item_features, user_features, cooc_scores,
    )

    train_df.to_parquet(results_dir / "train_feature_matrix.parquet", index=False)
    val_df.to_parquet(results_dir / "val_feature_matrix.parquet", index=False)
    logger.info("Feature matrices saved")

    logger.info("Training ranker")
    ranker = LightGBMRanker()
    ranker.fit(train_df, val_df)

    model_path = str(models_dir / "ranker.lgb")
    ranker.save(model_path)

    importance = ranker.feature_importance()
    importance.to_csv(results_dir / "feature_importance.csv", index=False)
    logger.info("Feature importance saved")
    print("\n--- Top 15 Features by Gain ---")
    print(importance.head(15).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()
    main(args.config)
