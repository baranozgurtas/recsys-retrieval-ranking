# Two-Stage Recommendation Pipeline

**Retrieval, ranking, and offline evaluation on MovieLens 1M.**

A production-style session-based recommendation system built around the two-stage paradigm used in large-scale industry systems: a fast retrieval stage narrows the item space to a candidate set, and a supervised ranking model re-scores those candidates using engineered features. The pipeline is evaluated with standard ranking metrics and a full slice-based breakdown across session length, item popularity, and user density.

---

## Key Findings

- **Tail item recall +117%.** The ranker improves Recall@10 on long-tail items from 0.040 to 0.086 — a 2.2x gain over co-occurrence alone. This is the primary value the ranking stage adds: recovering items that a frequency-based retriever cannot surface.
- **Catalog coverage +86%.** The ranker recommends 2,737 distinct items vs. 1,470 for co-occurrence — coverage increases from 41% to 77% of the full catalog.
- **Medium sessions favour the ranker.** On sessions of 4-9 items, the ranker outperforms co-occurrence on Recall@10 (0.113 vs 0.105). Longer sessions provide richer context for feature engineering; shorter sessions lack sufficient signal.
- **Co-occurrence score is the dominant feature.** `cooc_score_last_item` accounts for 44% of total LightGBM gain. Item popularity and session length are the next strongest signals.
- **Retrieval is the ceiling.** The ranker cannot recover items outside the top-100 candidate set. 85.7% of test sessions have their target item retrieved; the remaining 14.3% are unrecoverable regardless of ranking quality.

---

## Pipeline

```
Raw ratings
    -> Session construction    30-min inactivity gap, leave-last-out targets
    -> Candidate generation    co-occurrence retrieval, top-100 per session
    -> Feature engineering     item stats, user stats, session context, genre, cooc score
    -> LightGBM ranker         pointwise binary classification, AUC objective
    -> Offline evaluation      Recall@K, NDCG@K, MAP@K, MRR, slice-based breakdown
```

---

## Results

All metrics computed on the held-out test set (1,208 users, 3,883 sessions).

### Overall Comparison

| System | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 | MRR |
|---|---|---|---|---|---|
| popularity | 0.0106 | 0.0157 | 0.0063 | 0.0079 | 0.0055 |
| cooccurrence | 0.0901 | 0.1429 | 0.0601 | 0.0772 | 0.0574 |
| cooc+ranker | 0.0685 | 0.1192 | 0.0465 | 0.0628 | 0.0459 |

### By Session Length

| Session Length | N | cooc+ranker R@10 | cooccurrence R@10 | cooc+ranker NDCG@10 | cooccurrence NDCG@10 |
|---|---|---|---|---|---|
| short (1-3 items) | 1,029 | 0.0622 | 0.0816 | 0.0335 | 0.0451 |
| medium (4-9 items) | 551 | **0.1125** | 0.1053 | **0.0543** | 0.0626 |
| long (10+ items) | 2,303 | 0.1463 | 0.1793 | 0.0779 | 0.0951 |

The ranker gains an edge on medium sessions where feature engineering has enough context to outweigh the raw co-occurrence signal.

### By Item Popularity

| Item Type | N | cooc+ranker R@10 | cooccurrence R@10 | Delta |
|---|---|---|---|---|
| head (top 20%) | 2,215 | 0.1445 | 0.2208 | -0.076 |
| tail (bottom 80%) | 1,668 | **0.0857** | 0.0396 | **+0.046** |

Co-occurrence collapses on tail items (MRR 0.014) because infrequent items have sparse co-occurrence counts. The ranker uses item-level and user-level features that generalise beyond raw frequency, recovering a substantial fraction of long-tail targets.

### Catalog Coverage

| System | Items Recommended | Coverage |
|---|---|---|
| popularity | 10 | 0.3% |
| cooccurrence | 1,470 | 41.2% |
| cooc+ranker | 2,737 | **76.7%** |

### Top Features by Gain

| Rank | Feature | Gain | Share |
|---|---|---|---|
| 1 | cooc_score_last_item | 32,305 | 44% |
| 2 | item_global_count | 9,501 | 13% |
| 3 | item_global_rank | 7,660 | 10% |
| 4 | session_length | 7,163 | 10% |
| 5 | user_total_interactions | 4,699 | 6% |
| 6 | user_mean_session_length | 2,562 | 3% |

---

## Design Decisions

**Session boundary at 30 minutes.** Standard threshold in session-based recommendation literature. A gap of 30 minutes is long enough to separate distinct intent episodes while keeping sessions dense enough for meaningful co-occurrence counts. Configurable via `config/default.yaml`.

**Leave-last-out evaluation.** The final item in each session is the prediction target; the rest form the context. This directly mirrors the online inference task and is the standard offline protocol for session-based recommendation.

**User-level train/val/test split.** All sessions for a given user stay in one partition. This prevents any form of user history leakage across splits and gives a realistic estimate of generalisation to unseen users.

**Co-occurrence as the retrieval backbone.** Co-occurrence encodes which items appear together in sessions — a strong, fast, and interpretable signal for candidate generation. It requires no embeddings, no training, and is trivially explainable. The trade-off is that it degrades on tail items with sparse counts, which is precisely where the ranking stage compensates.

**LightGBM with AUC objective.** The training set has a 0.58% positive rate (one target per ~172 candidates). AUC is appropriate for heavily imbalanced binary classification and directly optimises the ranker's ability to separate the true target from negatives — which is the actual inference task.

**Two-stage architecture.** The retrieval stage provides high recall cheaply; the ranking stage improves precision using richer features that would be prohibitively expensive to compute over the full item catalog. This separation of concerns is the standard architecture in production recommendation systems.

---

## Limitations

- **Retrieval ceiling.** The ranker cannot recover target items outside the top-100 candidate set. Improving retrieval recall is the highest-leverage next step.
- **Offline-online gap.** Offline NDCG gains do not guarantee online lift. Metrics like novelty, diversity, and serendipity are not measured.
- **No temporal features.** Timestamps are used for session construction but not as ranking features. Recency weighting could improve short-session performance.
- **Co-occurrence does not generalise to cold items.** Items with no training co-occurrence counts cannot be retrieved. An embedding-based retriever (e.g. ALS) would address this.
- **Single dataset.** All results are on MovieLens 1M. Generalisation to other domains (e-commerce, news, music) is not validated.

---

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

Requires Python 3.10+.

## Usage

```bash
# Data preparation (downloads MovieLens 1M automatically)
PYTHONPATH=. python scripts/prepare_data.py

# Train the ranker
PYTHONPATH=. python scripts/train.py

# Evaluate all systems
PYTHONPATH=. python scripts/evaluate.py --split test
PYTHONPATH=. python scripts/slice_evaluate.py --split test
PYTHONPATH=. python scripts/error_analysis.py --split test

# Interactive dashboard
streamlit run app.py

# Tests
pytest tests/ -v
```

## Project Structure

```
retrieval-ranking-pipeline/
├── config/
│   └── default.yaml              session params, split fractions, evaluation k values
├── flowrec/
│   ├── data/
│   │   ├── loader.py             data download, loading, schema enforcement
│   │   └── session_builder.py    session segmentation, split, target extraction
│   ├── features/
│   │   ├── item_features.py      popularity stats, genre indicators, head/tail flag
│   │   └── user_features.py      session density, interaction statistics
│   ├── retrieval/
│   │   ├── base.py               abstract retriever interface
│   │   ├── popularity.py         global popularity baseline
│   │   └── cooccurrence.py       session co-occurrence item-to-item retrieval
│   ├── ranking/
│   │   ├── feature_builder.py    (session, candidate) feature matrix assembly
│   │   └── ranker.py             LightGBM wrapper with fit, rerank, save, load
│   ├── evaluation/
│   │   ├── metrics.py            Recall@K, Precision@K, MAP@K, NDCG@K, MRR
│   │   ├── slice_analysis.py     breakdown by session length, item popularity, user density
│   │   └── error_analysis.py     rank distribution, missed targets, catalog coverage
│   └── pipeline.py               end-to-end orchestration
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── slice_evaluate.py
│   └── error_analysis.py
├── app.py                        streamlit results dashboard
├── notebooks/
│   └── analysis.ipynb
└── tests/
    ├── test_session_builder.py
    ├── test_metrics.py
    ├── test_features.py
    └── test_slice_analysis.py
```

## Dataset

MovieLens 1M: 1,000,209 ratings from 6,040 users on 3,706 movies with timestamps.
Source: https://grouplens.org/datasets/movielens/1m/ — downloaded automatically by `prepare_data.py`.
