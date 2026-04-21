from abc import ABC, abstractmethod

import pandas as pd


class BaseRetriever(ABC):
    """
    Abstract interface for all candidate generation strategies.

    All retrievers consume session context and return a ranked list
    of candidate item IDs per session.
    """

    @abstractmethod
    def fit(self, sessions: pd.DataFrame) -> None:
        """Train or build internal structures from session data."""

    @abstractmethod
    def retrieve(self, context: pd.DataFrame, top_k: int) -> dict[int, list[int]]:
        """
        Generate top_k candidates for each session.

        Parameters
        ----------
        context : session context rows (all items except the target)
        top_k   : number of candidates to return per session

        Returns
        -------
        dict mapping session_id -> list of candidate item_ids (ranked)
        """
