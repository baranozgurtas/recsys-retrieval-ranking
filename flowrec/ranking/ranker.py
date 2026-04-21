import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "session_length",
    "item_global_count",
    "item_global_rank",
    "item_unique_users",
    "item_unique_sessions",
    "item_is_head",
    "user_total_interactions",
    "user_total_sessions",
    "user_mean_session_length",
    "user_is_dense",
    "cooc_score_last_item",
]


def _get_feature_cols(df):
    genre_cols = [c for c in df.columns if c.startswith("genre_")]
    return [c for c in FEATURE_COLS if c in df.columns] + genre_cols


class LightGBMRanker:
    def __init__(self, params=None):
        import lightgbm as lgb
        self._lgb = lgb
        self.params = params or {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 20,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "lambda_l1": 0.05,
            "lambda_l2": 0.05,
            "verbose": -1,
            "n_jobs": -1,
        }
        self.num_boost_round = 500
        self.model = None
        self.feature_cols = []

    def fit(self, train_df, val_df=None):
        self.feature_cols = _get_feature_cols(train_df)
        X_train = train_df[self.feature_cols].values
        y_train = train_df["label"].values
        train_data = self._lgb.Dataset(X_train, label=y_train)
        valid_sets = [train_data]
        callbacks = [self._lgb.log_evaluation(period=50)]
        if val_df is not None:
            X_val = val_df[self.feature_cols].values
            y_val = val_df["label"].values
            val_data = self._lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            callbacks.append(self._lgb.early_stopping(stopping_rounds=30, verbose=True))
        logger.info("Training LightGBM ranker: %d samples, %d features, %d positive labels",
                    len(X_train), len(self.feature_cols), int(y_train.sum()))
        self.model = self._lgb.train(
            self.params, train_data,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            callbacks=callbacks,
        )
        logger.info("Training complete. Best iteration: %d", self.model.best_iteration)

    def predict(self, df):
        if self.model is None:
            raise RuntimeError("Model has not been trained.")
        return self.model.predict(df[self.feature_cols].values, num_iteration=self.model.best_iteration)

    def rerank(self, feature_df, top_k):
        scores = self.predict(feature_df)
        feature_df = feature_df.copy()
        feature_df["score"] = scores
        results = {}
        for sid, group in feature_df.groupby("session_id"):
            ranked = group.sort_values("score", ascending=False)
            results[int(sid)] = ranked["candidate_item_id"].tolist()[:top_k]
        return results

    def feature_importance(self):
        if self.model is None:
            raise RuntimeError("Model has not been trained.")
        importance = self.model.feature_importance(importance_type="gain")
        return (
            pd.DataFrame({"feature": self.feature_cols, "importance": importance})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def save(self, path):
        if self.model is None:
            raise RuntimeError("No model to save.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(path)
        with open(str(path) + ".feature_cols.json", "w") as f:
            json.dump(self.feature_cols, f)
        logger.info("Model saved to %s", path)

    def load(self, path):
        self.model = self._lgb.Booster(model_file=path)
        with open(str(path) + ".feature_cols.json") as f:
            self.feature_cols = json.load(f)
        logger.info("Model loaded from %s", path)
