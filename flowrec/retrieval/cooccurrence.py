import logging
from collections import defaultdict

import pandas as pd

from flowrec.retrieval.base import BaseRetriever

logger = logging.getLogger(__name__)


class CooccurrenceRetriever(BaseRetriever):
    """
    Item-to-item retrieval based on session co-occurrence counts.

    For each session, scores candidate items by summing co-occurrence
    counts with items in the session context. Items already in the
    context are excluded from candidates.
    """

    def __init__(self, context_window: int = 5) -> None:
        self.context_window = context_window
        self._cooc: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._all_items: list[int] = []

    def fit(self, sessions: pd.DataFrame) -> None:
        self._cooc.clear()

        for sid, group in sessions.groupby("session_id"):
            items = group.sort_values("position")["item_id"].tolist()
            for i, item_a in enumerate(items):
                window = items[max(0, i - self.context_window) : i + self.context_window + 1]
                for item_b in window:
                    if item_b != item_a:
                        self._cooc[item_a][item_b] += 1

        self._all_items = sessions["item_id"].unique().tolist()
        logger.info(
            "CooccurrenceRetriever fitted: %d items, %d item pairs",
            len(self._all_items),
            sum(len(v) for v in self._cooc.values()),
        )

    def retrieve(self, context: pd.DataFrame, top_k: int) -> dict[int, list[int]]:
        results: dict[int, list[int]] = {}

        for sid, group in context.groupby("session_id"):
            session_items = set(group["item_id"].tolist())
            recent = (
                group.sort_values("position", ascending=False)
                .head(self.context_window)["item_id"]
                .tolist()
            )

            scores: dict[int, float] = defaultdict(float)
            for item in recent:
                for neighbor, count in self._cooc.get(item, {}).items():
                    if neighbor not in session_items:
                        scores[neighbor] += count

            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            results[sid] = [item for item, _ in ranked[:top_k]]

        return results
