"""
End-to-end pipeline entry point.

Runs the full sequence: data preparation -> training -> evaluation -> slices -> error analysis.
Intended for reproducibility. Individual scripts can be run independently for partial runs.

Usage:
    python -m flowrec.pipeline
    python -m flowrec.pipeline --config config/default.yaml --split val
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

SCRIPTS = [
    ("prepare_data",   ["python", "scripts/prepare_data.py"]),
    ("train",          ["python", "scripts/train.py"]),
    ("evaluate",       ["python", "scripts/evaluate.py"]),
    ("slice_evaluate", ["python", "scripts/slice_evaluate.py"]),
    ("error_analysis", ["python", "scripts/error_analysis.py"]),
]


def run_step(name: str, cmd: list[str], config: str, split: str) -> None:
    full_cmd = cmd + ["--config", config]
    if name not in ("prepare_data", "train"):
        full_cmd += ["--split", split]

    logger.info("Running step: %s", name)
    result = subprocess.run(full_cmd, check=False)
    if result.returncode != 0:
        logger.error("Step %s failed with exit code %d", name, result.returncode)
        sys.exit(result.returncode)
    logger.info("Step %s complete.", name)


def main(config: str, split: str) -> None:
    for name, cmd in SCRIPTS:
        run_step(name, cmd, config, split)
    logger.info("Full pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    args = parser.parse_args()
    main(args.config, args.split)
