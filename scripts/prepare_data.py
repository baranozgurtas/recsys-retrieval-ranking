"""
Download and preprocess the MovieLens 1M dataset.

Outputs (written to data/processed/):
    sessions.parquet         - all sessions, globally unique session_id
    train_sessions.parquet
    val_sessions.parquet
    test_sessions.parquet
    train_context.parquet    - train sessions with last item removed
    train_targets.parquet    - last item per train session
    val_context.parquet
    val_targets.parquet
    test_context.parquet
    test_targets.parquet
    movies.parquet

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --config config/default.yaml
"""

import argparse
import logging
from pathlib import Path

from flowrec.data.loader import (
    download_movielens_1m,
    load_config,
    load_movies,
    load_ratings,
)
from flowrec.data.session_builder import (
    build_sessions,
    extract_targets,
    split_sessions,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


def main(config_path: str) -> None:
    cfg = load_config(config_path)

    raw_dir = cfg["data"]["raw_dir"]
    processed_dir = Path(cfg["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    download_movielens_1m(raw_dir)

    ratings = load_ratings(raw_dir, separator=cfg["data"]["separator"])
    movies = load_movies(raw_dir, separator=cfg["data"]["separator"])

    movies.to_parquet(processed_dir / "movies.parquet", index=False)
    logger.info("Saved movies.parquet")

    sess_cfg = cfg["session"]
    sessions = build_sessions(
        ratings,
        gap_seconds=sess_cfg["gap_seconds"],
        min_length=sess_cfg["min_session_length"],
        max_length=sess_cfg["max_session_length"],
    )
    sessions.to_parquet(processed_dir / "sessions.parquet", index=False)
    logger.info("Saved sessions.parquet")

    split_cfg = cfg["split"]
    train, val, test = split_sessions(
        sessions,
        test_users_fraction=split_cfg["test_users_fraction"],
        val_users_fraction=split_cfg["val_users_fraction"],
        random_seed=split_cfg["random_seed"],
    )

    for name, split in [("train", train), ("val", val), ("test", test)]:
        split.to_parquet(processed_dir / f"{name}_sessions.parquet", index=False)
        logger.info("Saved %s_sessions.parquet (%d rows)", name, len(split))

        context, targets = extract_targets(split)
        context.to_parquet(processed_dir / f"{name}_context.parquet", index=False)
        targets.to_parquet(processed_dir / f"{name}_targets.parquet", index=False)
        logger.info(
            "Saved %s_context.parquet and %s_targets.parquet", name, name
        )

    logger.info("Phase 1 complete. All outputs written to %s", processed_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()
    main(args.config)
