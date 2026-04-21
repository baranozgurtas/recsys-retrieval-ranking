import logging
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def load_config(config_path: str = "config/default.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def download_movielens_1m(raw_dir: str) -> None:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if (raw_dir / "ratings.dat").exists():
        logger.info("MovieLens 1M already present, skipping download.")
        return

    zip_path = raw_dir / "ml-1m.zip"
    logger.info("Downloading MovieLens 1M from %s", _MOVIELENS_1M_URL)
    urllib.request.urlretrieve(_MOVIELENS_1M_URL, zip_path)

    logger.info("Extracting to %s", raw_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            filename = Path(member).name
            if not filename:
                continue
            with zf.open(member) as source:
                with open(raw_dir / filename, "wb") as target:
                    target.write(source.read())

    zip_path.unlink()
    logger.info("Download complete.")


def load_ratings(raw_dir: str, separator: str = "::") -> pd.DataFrame:
    path = Path(raw_dir) / "ratings.dat"
    df = pd.read_csv(
        path,
        sep=separator,
        engine="python",
        names=["user_id", "item_id", "rating", "timestamp"],
        dtype={"user_id": int, "item_id": int, "rating": float, "timestamp": int},
    )
    logger.info("Loaded %d ratings", len(df))
    return df


def load_movies(raw_dir: str, separator: str = "::") -> pd.DataFrame:
    path = Path(raw_dir) / "movies.dat"
    df = pd.read_csv(
        path,
        sep=separator,
        engine="python",
        names=["item_id", "title", "genres"],
        dtype={"item_id": int, "title": str, "genres": str},
        encoding="latin-1",
    )
    df["genre_list"] = df["genres"].str.split("|")
    logger.info("Loaded %d movies", len(df))
    return df


def load_users(raw_dir: str, separator: str = "::") -> pd.DataFrame:
    path = Path(raw_dir) / "users.dat"
    df = pd.read_csv(
        path,
        sep=separator,
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        dtype={"user_id": int},
    )
    logger.info("Loaded %d users", len(df))
    return df
