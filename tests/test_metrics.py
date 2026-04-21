import pytest
from flowrec.evaluation.metrics import (
    average_precision_at_k,
    evaluate_recommendations,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_perfect():
    assert recall_at_k([1, 2, 3], {1, 2, 3}, k=3) == 1.0


def test_recall_zero():
    assert recall_at_k([4, 5, 6], {1, 2, 3}, k=3) == 0.0


def test_recall_partial():
    assert recall_at_k([1, 4, 5], {1, 2, 3}, k=3) == pytest.approx(1 / 3)


def test_recall_truncates_at_k():
    assert recall_at_k([1, 2, 3], {3}, k=2) == 0.0


def test_recall_empty_relevant():
    assert recall_at_k([1, 2], set(), k=2) == 0.0


def test_precision_at_k():
    assert precision_at_k([1, 2, 3, 4], {1, 3}, k=4) == pytest.approx(2 / 4)


def test_precision_no_hits():
    assert precision_at_k([4, 5], {1, 2}, k=2) == 0.0


def test_ap_single_relevant_first():
    assert average_precision_at_k([1, 2, 3], {1}, k=3) == pytest.approx(1.0)


def test_ap_single_relevant_second():
    assert average_precision_at_k([2, 1, 3], {1}, k=3) == pytest.approx(0.5)


def test_ap_empty_relevant():
    assert average_precision_at_k([1, 2], set(), k=2) == 0.0


def test_ndcg_perfect():
    assert ndcg_at_k([1, 2, 3], {1, 2, 3}, k=3) == pytest.approx(1.0)


def test_ndcg_no_hits():
    assert ndcg_at_k([4, 5], {1, 2}, k=2) == 0.0


def test_ndcg_partial():
    score = ndcg_at_k([2, 1], {1}, k=2)
    assert 0.0 < score < 1.0


def test_mrr_first_hit():
    assert reciprocal_rank([1, 2, 3], {1}) == pytest.approx(1.0)


def test_mrr_second_hit():
    assert reciprocal_rank([2, 1, 3], {1}) == pytest.approx(0.5)


def test_mrr_no_hit():
    assert reciprocal_rank([2, 3], {1}) == 0.0


def test_evaluate_recommendations_aggregates():
    recommendations = {
        0: [1, 2, 3, 4, 5],
        1: [5, 6, 7, 8, 9],
    }
    targets = {
        0: {1},
        1: {5},
    }
    results = evaluate_recommendations(recommendations, targets, k_values=[5])
    assert results["recall@5"] == pytest.approx(1.0)
    assert results["mrr"] == pytest.approx(1.0)


def test_evaluate_recommendations_missing_session():
    recommendations = {0: [1, 2]}
    targets = {0: {1}, 1: {3}}
    results = evaluate_recommendations(recommendations, targets, k_values=[2])
    assert results["recall@2"] == pytest.approx(0.5)
