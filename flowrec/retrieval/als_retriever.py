import logging

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from flowrec.retrieval.base import BaseRetriever

logger = logging.getLogger(__name__)


class ALSRetriever(BaseRetriever):
    """
    Embedding-based retrieval using Alternating Least Squares (ALS).

    Requires the `implicit` library:
        pip install implicit

    Note: implicit does not currently support Python 3.13.
    Use Python 3.10 or 3.11 if this retriever is needed.
    """

    def __init__(self, factors: int = 64, iterations: int = 20, regularization: float = 0.01) -> None:
        try:
            import implicit
            self._implicit = implicit
        except ImportError as e:
            raise ImportError(
                "The 'implicit' library is required for ALSRetriever. "
                "Install it with: pip install implicit\n"
                "Note: implicit does not support Python 3.13. Use Python 3.10 or 3.11."
            ) from e

        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self._model = None
        self._item_ids: list[int] = []
        self._item_index: dict[int, int] = {}

    def fit(self, sessions: pd.DataFrame) -> None:
        self._item_ids = sorted(sessions["item_id"].unique().tolist())
        self._item_index = {item: i for i, item in enumerate(self._item_ids)}
        user_ids = sorted(sessions["user_id"].unique().tolist())
        user_index = {u: i for i, u in enumerate(user_ids)}

        rows = sessions["item_id"].map(self._item_index).values
        cols = sessions["user_id"].map(user_index).values
        data = np.ones(len(sessions), dtype=np.float32)

        item_user = csr_matrix(
            (data, (rows, cols)),
            shape=(len(self._item_ids), len(user_ids)),
        )

        self._model = self._implicit.als.AlternatingLeastSquares(
            factors=self.factors,
            iterations=self.iterations,
            regularization=self.regularization,
        )
        self._model.fit(item_user)
        logger.info("ALSRetriever fitted: %d items, %d users", len(self._item_ids), len(user_ids))

    def retrieve(self, context: pd.DataFrame, top_k: int) -> dict[int, list[int]]:
        if self._model is None:
            raise RuntimeError("Model has not been fitted.")

        results: dict[int, list[int]] = {}
        for sid, group in context.groupby("session_id"):
            session_items = group["item_id"].tolist()
            known = [self._item_index[i] for i in session_items if i in self._item_index]
            if not known:
                results[sid] = []
                continue

            item_vectors = self._model.item_factors[known]
            session_vec = item_vectors.mean(axis=0)
            scores = self._model.item_factors @ session_vec
            top_indices = np.argsort(-scores)

            session_item_set = set(session_items)
            candidates = [
                self._item_ids[i]
                for i in top_indices
                if self._item_ids[i] not in session_item_set
            ][:top_k]
            results[sid] = candidates

        return results
