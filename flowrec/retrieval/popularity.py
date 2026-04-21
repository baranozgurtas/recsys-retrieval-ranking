import logging

import pandas as pd

from flowrec.retrieval.base import BaseRetriever

logger = logging.getLogger(__name__)


class PopularityRetriever(BaseRetriever):
    """
    Retrieves the globally most popular items regardless of session context.

    This serves as the primary baseline. An effective ranking model must
    outperform it on tail items and personalised sessions.
    """

    def __init__(self) -> None:
        self._top_items: list[int] = []

    def fit(self, sessions: pd.DataFrame) -> None:
        counts = sessions["item_id"].value_counts()
        self._top_items = counts.index.tolist()
        logger.info("PopularityRetriever fitted on %d unique items", len(self._top_items))

    def retrieve(self, context: pd.DataFrame, top_k: int) -> dict[int, list[int]]:
        candidates = self._top_items[:top_k]
        session_ids = context["session_id"].unique()
        return {sid: candidates for sid in session_ids}
