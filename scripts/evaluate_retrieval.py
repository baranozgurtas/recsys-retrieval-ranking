"""
Evaluate candidate generation baselines on the validation set.

Outputs:
    results/retrieval_eval.csv   - metrics per retriever per k

Usage:
    python scripts/evaluate_retrieval.py
    python scripts/evaluate_retrieval.py --config config/default.yaml --split val
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from flowrec.data.loader import load_config
from flowrec.evaluation.metrics import evaluate_recommendations
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

    top_k = cfg["retrieval"]["top_k"]
    k_values = cfg["evaluation"]["k_values"]

    logger.info("Loading data for split: %s", split)
    train_sessions = pd.read_parquet(processed_dir / "train_sessions.parquet")
    context = pd.read_parquet(processed_dir / f"{split}_context.parquet")
    targets_df = pd.read_parquet(processed_dir / f"{split}_targets.parquet")
    targets = build_target_dict(targets_df)

    retrievers = {
        "popularity": PopularityRetriever(),
        "cooccurrence": CooccurrenceRetriever(context_window=5),
    }

    rows = []
    for name, retriever in retrievers.items():
        logger.info("Fitting %s", name)
        retriever.fit(train_sessions)

        logger.info("Retrieving candidates with %s (top_k=%d)", name, top_k)
        recommendations = retriever.retrieve(context, top_k=top_k)

        logger.info("Evaluating %s", name)
        metrics = evaluate_recommendations(recommendations, targets, k_values)

        for metric, value in metrics.items():
            rows.append({"retriever": name, "metric": metric, "value": round(value, 5)})
            logger.info("  %s %s=%.5f", name, metric, value)

    results_df = pd.DataFrame(rows)
    out_path = results_dir / "retrieval_eval.csv"
    results_df.to_csv(out_path, index=False)
    logger.info("Results written to %s", out_path)

    # print summary table
    pivot = results_df.pivot(index="retriever", columns="metric", values="value")
    print("\n--- Retrieval Evaluation Summary ---")
    print(pivot.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    args = parser.parse_args()
    main(args.config, args.split)
