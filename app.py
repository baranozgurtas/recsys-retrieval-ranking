"""
FlowRec — Results Dashboard

Reads pre-computed results from results/ and renders an interactive summary.

Usage:
    streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

RESULTS = Path("results")

st.set_page_config(
    page_title="FlowRec",
    page_icon=None,
    layout="wide",
)

st.title("FlowRec — Session-Based Recommendation Pipeline")
st.caption(
    "Two-stage retrieve-then-rank system trained on MovieLens 1M. "
    "All results computed on the held-out test set."
)

# ── helpers ──────────────────────────────────────────────────────────────────

def load_csv(name: str) -> pd.DataFrame | None:
    path = RESULTS / name
    if not path.exists():
        return None
    return pd.read_csv(path)


def check_results() -> bool:
    return (RESULTS / "final_eval.csv").exists()


# ── sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.header("Navigation")
section = st.sidebar.radio(
    "Section",
    ["Overview", "System Comparison", "Slice Analysis", "Feature Importance", "Error Analysis"],
)

if not check_results():
    st.error(
        "No results found. Run the pipeline first:\n\n"
        "```\nPYTHONPATH=. python scripts/train.py\n"
        "PYTHONPATH=. python scripts/evaluate.py --split test\n"
        "PYTHONPATH=. python scripts/slice_evaluate.py --split test\n"
        "PYTHONPATH=. python scripts/error_analysis.py --split test\n```"
    )
    st.stop()

# ── overview ─────────────────────────────────────────────────────────────────

if section == "Overview":
    st.header("Pipeline Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset", "MovieLens 1M")
    col2.metric("Total Sessions", "18,284")
    col3.metric("Items", "3,567")

    col4, col5, col6 = st.columns(3)
    col4.metric("Train Sessions", "12,466")
    col5.metric("Val Sessions", "1,935")
    col6.metric("Test Sessions", "3,883")

    st.subheader("Architecture")
    st.code(
        "Raw ratings\n"
        "    → Session construction   (30-min inactivity gap)\n"
        "    → Candidate generation   (co-occurrence, top-100)\n"
        "    → Feature engineering    (item stats, user stats, genre, cooc score)\n"
        "    → LightGBM ranker        (pointwise, scale_pos_weight)\n"
        "    → Offline evaluation     (Recall@K, NDCG@K, MRR)",
        language=None,
    )

    st.subheader("Systems Compared")
    st.table(pd.DataFrame({
        "System": ["popularity", "cooccurrence", "cooc+ranker"],
        "Description": [
            "Global top-N items, no personalisation",
            "Items co-occurring in sessions with context items",
            "Co-occurrence candidates re-ranked by LightGBM",
        ]
    }))

# ── system comparison ─────────────────────────────────────────────────────────

elif section == "System Comparison":
    st.header("System Comparison")

    eval_df = load_csv("final_eval.csv")
    if eval_df is None:
        st.warning("Run evaluate.py first.")
        st.stop()

    pivot = eval_df.pivot(index="system", columns="metric", values="value")
    ordered = ["recall@5", "recall@10", "recall@20", "ndcg@5", "ndcg@10", "ndcg@20",
               "map@5", "map@10", "map@20", "mrr", "precision@5", "precision@10", "precision@20"]
    ordered = [c for c in ordered if c in pivot.columns]
    pivot = pivot[ordered]

    st.subheader("Full Metrics Table")
    st.dataframe(pivot.style.format("{:.4f}").highlight_max(axis=0, color="#d4edda"))

    st.subheader("Key Metrics")
    plot_metrics = ["recall@10", "ndcg@10", "mrr"]
    plot_data = eval_df[eval_df["metric"].isin(plot_metrics)]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    colors = ["#4e79a7", "#f28e2b", "#59a14f"]
    for ax, metric in zip(axes, plot_metrics):
        subset = plot_data[plot_data["metric"] == metric].sort_values("value")
        ax.barh(subset["system"], subset["value"], color=colors[:len(subset)])
        ax.set_title(metric, fontweight="bold")
        ax.set_xlabel("Score")
        for i, (_, row) in enumerate(subset.iterrows()):
            ax.text(row["value"] + 0.001, i, f"{row['value']:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── slice analysis ────────────────────────────────────────────────────────────

elif section == "Slice Analysis":
    st.header("Slice Analysis")

    slice_df = load_csv("slice_eval_combined.csv")
    if slice_df is None:
        st.warning("Run slice_evaluate.py first.")
        st.stop()

    metric = st.selectbox("Metric", ["recall@10", "ndcg@10", "mrr", "recall@20"])
    slice_type = st.selectbox("Slice", slice_df["slice_type"].unique().tolist())

    subset = slice_df[
        (slice_df["slice_type"] == slice_type) &
        (slice_df["metric"] == metric)
    ]

    if subset.empty:
        st.warning("No data for this selection.")
        st.stop()

    pivot = subset.pivot(index="slice_value", columns="system", values="value")

    st.subheader(f"{metric} by {slice_type}")
    st.dataframe(pivot.style.format("{:.4f}").highlight_max(axis=1, color="#d4edda"))

    fig, ax = plt.subplots(figsize=(8, 4))
    pivot.plot(kind="bar", ax=ax, colormap="tab10")
    ax.set_title(f"{metric} by {slice_type}", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel(metric)
    ax.legend(title="System", bbox_to_anchor=(1, 1))
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Key Observations")
    st.info(
        "- Ranker outperforms co-occurrence on **tail items** (long-tail problem)\n"
        "- Co-occurrence dominates on **head items** (strong co-occurrence signal)\n"
        "- Both systems degrade significantly on **short sessions** (sparse context)\n"
        "- Ranker provides higher **catalog coverage** across all session types"
    )

# ── feature importance ────────────────────────────────────────────────────────

elif section == "Feature Importance":
    st.header("Feature Importance")

    imp = load_csv("feature_importance.csv")
    if imp is None:
        st.warning("Run train.py first.")
        st.stop()

    top_n = st.slider("Top N features", min_value=5, max_value=len(imp), value=15)
    imp_top = imp.head(top_n)

    fig, ax = plt.subplots(figsize=(8, top_n * 0.4 + 1))
    ax.barh(imp_top["feature"][::-1], imp_top["importance"][::-1], color="#4e79a7")
    ax.set_title("Feature Importance (Gain)", fontweight="bold")
    ax.set_xlabel("Gain")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Full Table")
    st.dataframe(imp)

# ── error analysis ────────────────────────────────────────────────────────────

elif section == "Error Analysis":
    st.header("Error Analysis")

    cov = load_csv("coverage_stats.csv")
    if cov is not None:
        st.subheader("Catalog Coverage")
        st.dataframe(cov.set_index("system").style.format({"coverage": "{:.3f}"}))

    system = st.selectbox("System", ["cooc_ranker", "cooccurrence", "popularity"])
    report = load_csv(f"error_report_{system}.csv")

    if report is None:
        st.warning(f"No error report for {system}. Run error_analysis.py first.")
        st.stop()

    st.subheader("Target Rank Distribution")
    hits = report[report["target_rank"] > 0]["target_rank"]
    miss_pct = 100 * (report["target_rank"] == -1).mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Hit@10", f"{(report['hit_at_k'].mean()):.3f}")
    col2.metric("Missed (not retrieved)", f"{miss_pct:.1f}%")
    col3.metric("Median rank (when found)", f"{hits.median():.0f}" if len(hits) > 0 else "N/A")

    if len(hits) > 0:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.hist(hits, bins=20, color="#4e79a7", edgecolor="white", linewidth=0.5)
        ax.set_title(f"Target Item Rank Distribution — {system}", fontweight="bold")
        ax.set_xlabel("Rank")
        ax.set_ylabel("Sessions")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.subheader("Session Length vs Hit Rate")
    report["length_bucket"] = pd.cut(
        report["session_length"],
        bins=[0, 3, 9, 50],
        labels=["short (1-3)", "medium (4-9)", "long (10+)"],
    )
    bucket_stats = report.groupby("length_bucket", observed=True)["hit_at_k"].mean().reset_index()
    bucket_stats.columns = ["Session Length", "Hit@10"]
    st.dataframe(bucket_stats.style.format({"Hit@10": "{:.3f}"}))
