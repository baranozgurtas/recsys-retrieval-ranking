import numpy as np


def _check_inputs(recommended: list, relevant: set, k: int) -> None:
    if not isinstance(recommended, list):
        raise TypeError("recommended must be a list")
    if not isinstance(relevant, set):
        raise TypeError("relevant must be a set")
    if k < 1:
        raise ValueError("k must be >= 1")


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k


def average_precision_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for i, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1
            precision_sum += hits / i
    return precision_sum / min(len(relevant), k)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0

    def dcg(items):
        return sum(
            1.0 / np.log2(i + 2)
            for i, item in enumerate(items)
            if item in relevant
        )

    ideal = sorted([1 if item in relevant else 0 for item in recommended[:k]], reverse=True)
    ideal_items = [i for i, v in enumerate(ideal) if v == 1]
    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in ideal_items)

    if ideal_dcg == 0.0:
        return 0.0
    return dcg(recommended[:k]) / ideal_dcg


def reciprocal_rank(recommended: list, relevant: set) -> float:
    for i, item in enumerate(recommended, start=1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def evaluate_recommendations(
    recommendations: dict[int, list],
    targets: dict[int, set],
    k_values: list[int],
) -> dict[str, float]:
    """
    Compute mean ranking metrics over all sessions.

    Parameters
    ----------
    recommendations : {session_id: [ranked item ids]}
    targets         : {session_id: {relevant item ids}}
    k_values        : list of cutoff values, e.g. [5, 10, 20]

    Returns
    -------
    dict mapping metric names to their mean values across sessions
    """
    session_ids = list(targets.keys())
    results: dict[str, list[float]] = {
        f"recall@{k}": [] for k in k_values
    }
    results.update({f"precision@{k}": [] for k in k_values})
    results.update({f"map@{k}": [] for k in k_values})
    results.update({f"ndcg@{k}": [] for k in k_values})
    results["mrr"] = []

    for sid in session_ids:
        recs = recommendations.get(sid, [])
        relevant = targets[sid]

        results["mrr"].append(reciprocal_rank(recs, relevant))
        for k in k_values:
            results[f"recall@{k}"].append(recall_at_k(recs, relevant, k))
            results[f"precision@{k}"].append(precision_at_k(recs, relevant, k))
            results[f"map@{k}"].append(average_precision_at_k(recs, relevant, k))
            results[f"ndcg@{k}"].append(ndcg_at_k(recs, relevant, k))

    return {metric: float(np.mean(values)) for metric, values in results.items()}
