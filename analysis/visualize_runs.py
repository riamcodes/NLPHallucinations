"""Generate visual summaries of QA run logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import fill
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

RUNS_DIR = Path("runs")
PLOTS_DIR = RUNS_DIR / "plots"


def find_latest_run(run_dir: Path) -> Path:
    """Return the most recent JSONL run file."""
    candidates = sorted(run_dir.glob("run-*.jsonl"))
    if not candidates:
        raise FileNotFoundError("No run files found in 'runs/'.")
    return candidates[-1]


def load_run(path: Path) -> pd.DataFrame:
    """Load a JSONL run file into a DataFrame."""
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"Run file {path} is empty.")
    return pd.DataFrame(records)


def unpack_detectors(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten detector information into explicit columns."""
    def get_detector(row: Dict[str, Any], detector: str) -> Dict[str, Any]:
        return (row.get("detections") or {}).get(detector, {}) or {}

    def get_detector_value(
        row: Dict[str, Any], detector: str, key: str, default: Optional[Any] = None
    ) -> Any:
        return get_detector(row, detector).get(key, default)

    df = df.copy()

    # Context overlap
    df["context_overlap_score"] = df.apply(
        lambda row: get_detector_value(row, "context_overlap", "score"), axis=1
    )
    df["context_overlap_label"] = df.apply(
        lambda row: get_detector_value(row, "context_overlap", "label"), axis=1
    )

    # Self-consistency
    df["self_consistency_score"] = df.apply(
        lambda row: get_detector_value(row, "self_consistency", "score"), axis=1
    )
    df["self_consistency_label"] = df.apply(
        lambda row: get_detector_value(row, "self_consistency", "label"), axis=1
    )
    df["self_consistency_unique"] = df.apply(
        lambda row: get_detector_value(
            row,
            "self_consistency",
            "details",
            {},
        ).get("unique_replies", []),
        axis=1,
    )

    # Temperature sensitivity
    df["temperature_sensitivity_score"] = df.apply(
        lambda row: get_detector_value(row, "temperature_sensitivity", "score"), axis=1
    )
    df["temperature_sensitivity_label"] = df.apply(
        lambda row: get_detector_value(row, "temperature_sensitivity", "label"), axis=1
    )

    # Context ablation (new)
    df["context_ablation_score"] = df.apply(
        lambda row: get_detector_value(row, "context_ablation", "score"), axis=1
    )
    df["context_ablation_label"] = df.apply(
        lambda row: get_detector_value(row, "context_ablation", "label"), axis=1
    )

    # Any detector flagged? (treat context_ablation 'flag_context_ignored' as flag)
    df["flagged"] = df.apply(
        lambda row: any(
            label == "flag"
            or label == "flag_context_ignored"
            for label in (
                row.get("context_overlap_label"),
                row.get("self_consistency_label"),
                row.get("context_ablation_label"),
            )
            if label
        ),
        axis=1,
    )
    return df


def plot_detector_counts(df: pd.DataFrame, output_path: Path) -> None:
    # Check if we have multiple models
    has_model_col = "model" in df.columns and df["model"].nunique() > 1
    
    # Melt the data
    id_vars = ["model"] if has_model_col else []
    melted = pd.melt(
        df,
        id_vars=id_vars,
        value_vars=[
            "context_overlap_label",
            "self_consistency_label",
            "context_ablation_label",
        ],
        var_name="detector",
        value_name="label",
    )
    
    # Group by model (if present), detector, and label to get counts
    groupby_cols = id_vars + ["detector", "label"]
    counts = (
        melted.groupby(groupby_cols, dropna=False)
        .size()
        .reset_index(name="count")
    )

    counts["detector"] = counts["detector"].str.replace("_label", "", regex=False)
    sns.set_theme(style="whitegrid")
    
    if has_model_col:
        # Create subplots for each model
        models = sorted(df["model"].unique())
        fig, axes = plt.subplots(1, len(models), figsize=(9 * len(models), 4), sharey=True)
        if len(models) == 1:
            axes = [axes]
        
        for idx, model in enumerate(models):
            model_counts = counts[counts["model"] == model]
            sns.barplot(data=model_counts, x="detector", y="count", hue="label", palette="deep", ax=axes[idx])
            axes[idx].set_title(f"Detector label counts - {model}")
            axes[idx].set_ylabel("Count" if idx == 0 else "")
            axes[idx].set_xlabel("Detector")
            if idx > 0:
                axes[idx].get_legend().remove()
            else:
                axes[idx].legend(title="Label", bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.figure(figsize=(9, 4))
        sns.barplot(data=counts, x="detector", y="count", hue="label", palette="deep")
        plt.title("Detector label counts")
        plt.ylabel("Count")
        plt.xlabel("Detector")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_overlap_vs_consistency(df: pd.DataFrame, output_path: Path) -> None:
    # Check if we have multiple models
    has_model_col = "model" in df.columns and df["model"].nunique() > 1
    
    sns.set_theme(style="whitegrid")
    jitter_rng = np.random.default_rng(42)
    df = df.copy()
    df["context_jitter"] = df["context_overlap_score"] + jitter_rng.uniform(-0.02, 0.02, len(df))
    df["consistency_jitter"] = df["self_consistency_score"] + jitter_rng.uniform(-0.02, 0.02, len(df))

    if has_model_col:
        # Create subplots for each model
        models = df["model"].unique()
        fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 6), sharex=True, sharey=True)
        if len(models) == 1:
            axes = [axes]
        
        for idx, model in enumerate(models):
            model_df = df[df["model"] == model]
            sns.scatterplot(
                data=model_df,
                x="context_jitter",
                y="consistency_jitter",
                hue="flagged",
                style="flagged",
                palette={True: "#d62728", False: "#1f77b4"},
                s=80,
                ax=axes[idx],
            )
            
            for _, row in model_df[model_df["flagged"]].iterrows():
                axes[idx].annotate(
                    f"Q{row['qid']}",
                    (row["context_jitter"], row["consistency_jitter"]),
                    textcoords="offset points",
                    xytext=(6, 6),
                    fontsize=9,
                    color="#d62728",
                )
            
            axes[idx].set_title(f"Context overlap vs. self-consistency - {model}")
            axes[idx].set_xlabel("Context overlap score (jittered)")
            axes[idx].set_ylabel("Self-consistency diversity (jittered)" if idx == 0 else "")
            axes[idx].set_xlim(-0.05, 1.05)
            axes[idx].set_ylim(-0.05, 1.05)
            if idx > 0:
                axes[idx].get_legend().remove()
    else:
        plt.figure(figsize=(7, 6))
        sns.scatterplot(
            data=df,
            x="context_jitter",
            y="consistency_jitter",
            hue="flagged",
            style="flagged",
            palette={True: "#d62728", False: "#1f77b4"},
            s=80,
        )

        for _, row in df[df["flagged"]].iterrows():
            plt.annotate(
                f"Q{row['qid']}",
                (row["context_jitter"], row["consistency_jitter"]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=9,
                color="#d62728",
            )

        plt.title("Context overlap vs. self-consistency")
        plt.xlabel("Context overlap score (jittered)")
        plt.ylabel("Self-consistency diversity (jittered)")
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_flagged_question_bars(df: pd.DataFrame, output_path: Path) -> None:
    # Check if we have multiple models
    has_model_col = "model" in df.columns and df["model"].nunique() > 1
    
    df = df.copy()
    df["flag_count"] = df.apply(
        lambda row: sum(
            (
                row.get("context_overlap_label") == "flag",
                row.get("self_consistency_label") == "flag",
                row.get("context_ablation_label") == "flag_context_ignored",
            )
        ),
        axis=1,
    )
    
    sns.set_theme(style="whitegrid")
    
    if has_model_col:
        # Create subplots for each model
        models = sorted(df["model"].unique())
        fig, axes = plt.subplots(1, len(models), figsize=(9 * len(models), 12), sharey=True)
        if len(models) == 1:
            axes = [axes]
        
        for idx, model in enumerate(models):
            model_df = df[df["model"] == model].copy()
            # Sort by flag_count descending, then by qid for consistent ordering
            model_df = model_df.sort_values(["flag_count", "qid"], ascending=[False, True])
            
            # Create horizontal bar chart showing each question individually
            # Use matplotlib barh directly to avoid seaborn aggregation
            y_pos = range(len(model_df))
            axes[idx].barh(y_pos, model_df["flag_count"].values, color="#d62728")
            axes[idx].set_yticks(y_pos)
            axes[idx].set_yticklabels(model_df["qid"].values, fontsize=7)
            axes[idx].set_xlabel("Number of detectors flagging")
            axes[idx].set_ylabel("Question ID" if idx == 0 else "")
            axes[idx].set_title(f"Flagged detectors per question - {model}")
            axes[idx].set_xlim(0, 3.5)  # Max 3 detectors, with some padding
            axes[idx].invert_yaxis()  # Show highest flag count at top
    else:
        df = df.sort_values(["flag_count", "qid"], ascending=[False, True])
        plt.figure(figsize=(9, 12))
        y_pos = range(len(df))
        plt.barh(y_pos, df["flag_count"].values, color="#d62728")
        plt.yticks(y_pos, df["qid"].values, fontsize=7)
        plt.xlabel("Number of detectors flagging")
        plt.ylabel("Question ID")
        plt.title("Flagged detectors per question")
        plt.xlim(0, 3.5)
        plt.gca().invert_yaxis()  # Show highest flag count at top
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_flag_matrix(df: pd.DataFrame, output_path: Path) -> None:
    # Check if we have multiple models
    has_model_col = "model" in df.columns and df["model"].nunique() > 1
    
    cmap = ListedColormap(["#f0f0f0", "#d62728"])
    sns.set_theme(style="white")
    
    if has_model_col:
        # Create subplots for each model
        models = sorted(df["model"].unique())
        # Increase figure height to show all questions
        fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 14), sharey=True)
        if len(models) == 1:
            axes = [axes]
        
        for idx, model in enumerate(models):
            model_df = df[df["model"] == model].copy()
            # Sort by qid to ensure consistent ordering and reset index
            model_df = model_df.sort_values("qid").reset_index(drop=True)
            
            # Create flag columns - handle None/NaN values
            context_overlap_flags = (model_df["context_overlap_label"].fillna("") == "flag").astype(int)
            self_consistency_flags = (model_df["self_consistency_label"].fillna("") == "flag").astype(int)
            context_ablation_flags = (model_df["context_ablation_label"].fillna("") == "flag_context_ignored").astype(int)
            
            # Create matrix with qid as index
            matrix = pd.DataFrame(
                {
                    "context_overlap": context_overlap_flags.values,
                    "self_consistency": self_consistency_flags.values,
                    "context_ablation": context_ablation_flags.values,
                },
                index=model_df["qid"].values,
            )
            
            # Don't annotate for large matrices - too cluttered
            # But ensure all rows are visible
            sns.heatmap(
                matrix,
                cmap=cmap,
                linewidths=0.3,
                linecolor="#cccccc",
                cbar=False,
                annot=False,  # Disable annotation for clarity
                vmin=0,
                vmax=1,
                ax=axes[idx],
                yticklabels=True,  # Show all question IDs
                xticklabels=True,  # Show detector names
            )
            axes[idx].set_xlabel("Detector")
            axes[idx].set_ylabel("Question ID" if idx == 0 else "")
            axes[idx].set_title(f"Detector flags by question - {model}")
            # Set y-axis label rotation and font size
            plt.setp(axes[idx].get_yticklabels(), rotation=0, fontsize=7)
    else:
        df_sorted = df.sort_values("qid").reset_index(drop=True)
        # Fill NaN values before comparison
        context_overlap_flags = (df_sorted["context_overlap_label"].fillna("") == "flag").astype(int)
        self_consistency_flags = (df_sorted["self_consistency_label"].fillna("") == "flag").astype(int)
        context_ablation_flags = (df_sorted["context_ablation_label"].fillna("") == "flag_context_ignored").astype(int)
        
        matrix = pd.DataFrame(
            {
                "context_overlap": context_overlap_flags.values,
                "self_consistency": self_consistency_flags.values,
                "context_ablation": context_ablation_flags.values,
            },
            index=df_sorted["qid"].values,
        )

        plt.figure(figsize=(7, 14))
        sns.heatmap(
            matrix,
            cmap=cmap,
            linewidths=0.3,
            linecolor="#cccccc",
            cbar=False,
            annot=False,
            vmin=0,
            vmax=1,
            yticklabels=True,
        )
        plt.xlabel("Detector")
        plt.ylabel("Question ID")
        plt.title("Detector flags by question (red = flag)")
        plt.yticks(rotation=0, fontsize=7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_flagged_responses(df: pd.DataFrame, output_path: Path) -> None:
    flagged = df[df["self_consistency_label"] == "flag"]
    if flagged.empty:
        return

    lines = []
    for _, row in flagged.iterrows():
        question = fill(f"Q{row['qid']}: {row['question']}", width=90)
        baseline = fill(f"Baseline: {row['answer']}", width=90)
        samples = row["self_consistency_unique"] or []
        samples_wrapped = "\n".join(
            fill(f"- {sample}", width=90, subsequent_indent="  ")
            for sample in samples
        )
        block = f"{question}\n{baseline}\nStochastic samples:\n{samples_wrapped}"
        lines.append(block)

    text = "\n\n".join(lines)

    plt.figure(figsize=(10, max(6, 0.3 * len(text.splitlines()))))
    ax = plt.gca()
    ax.axis("off")
    ax.text(0, 1, text, va="top", ha="left", fontsize=10, family="monospace")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_temp_vs_self_consistency(df: pd.DataFrame, output_path: Path) -> None:
    """Create comprehensive comparison between temperature sensitivity and self-consistency."""
    has_model_col = "model" in df.columns and df["model"].nunique() > 1
    models = sorted(df["model"].unique()) if has_model_col else [None]
    
    # Filter to parametric knowledge questions (no context)
    df_filtered = df[(df["context"].isna()) | (df["context"].astype(str).str.strip() == "")]
    
    # Need to load raw data to get details
    def get_detector_details(row, detector_name):
        """Extract detector details from raw row data."""
        detections = row.get("detections", {})
        if isinstance(detections, dict):
            detector = detections.get(detector_name, {})
            return detector.get("details", {})
        return {}
    
    # Add sample count columns
    def get_sc_samples(row):
        details = get_detector_details(row, "self_consistency")
        return details.get("samples", 4)
    
    def get_temp_samples(row):
        details = get_detector_details(row, "temperature_sensitivity")
        temps = details.get("temperatures", [0.3, 0.7, 1.0])
        samples_per_temp = details.get("samples_per_temp", 2)
        return len(temps) * samples_per_temp
    
    # Load raw data to access details
    run_path = find_latest_run(RUNS_DIR)
    raw_records = []
    with run_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_records.append(json.loads(line))
    raw_df = pd.DataFrame(raw_records)
    
    # Merge sample counts
    df_filtered = df_filtered.copy()
    df_filtered["sc_samples"] = df_filtered.index.map(
        lambda idx: get_sc_samples(raw_df.iloc[idx]) if idx < len(raw_df) else 4
    )
    df_filtered["temp_samples"] = df_filtered.index.map(
        lambda idx: get_temp_samples(raw_df.iloc[idx]) if idx < len(raw_df) else 6
    )
    
    num_models = len(models)
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, num_models, hspace=0.3, wspace=0.3)
    
    for model_idx, model in enumerate(models):
        if model:
            df_model = df_filtered[df_filtered["model"] == model]
            title_suffix = f" - {model}"
        else:
            df_model = df_filtered
            title_suffix = ""
        
        if len(df_model) == 0:
            continue
        
        # 1. Scatter plot: Agreement between detectors
        ax1 = fig.add_subplot(gs[0, model_idx])
        sc_scores = df_model["self_consistency_score"].fillna(0)
        temp_scores = df_model["temperature_sensitivity_score"].fillna(0)
        
        # Color by agreement
        sc_flags = df_model["self_consistency_label"].str.startswith("flag", na=False)
        temp_flags = df_model["temperature_sensitivity_label"].str.startswith("flag", na=False)
        agreement = sc_flags == temp_flags
        
        colors = agreement.map({True: "green", False: "orange"}).fillna("gray")
        ax1.scatter(
            sc_scores, temp_scores,
            c=colors,
            alpha=0.6, s=50, edgecolors="black", linewidth=0.5
        )
        ax1.set_xlabel("Self-Consistency Score (diversity)", fontsize=10)
        ax1.set_ylabel("Temperature Sensitivity Score (mean drift)", fontsize=10)
        ax1.set_title(f"Detector Agreement{title_suffix}", fontsize=11, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.legend(
            [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, markeredgecolor='black'),
             plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=8, markeredgecolor='black')],
            ["Agree", "Disagree"],
            loc="upper left"
        )
        
        # Add correlation coefficient
        if len(sc_scores) > 1 and sc_scores.std() > 0 and temp_scores.std() > 0:
            corr = sc_scores.corr(temp_scores)
            ax1.text(0.05, 0.95, f"Correlation: {corr:.3f}", transform=ax1.transAxes,
                    fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 2. Performance comparison bar chart
        ax2 = fig.add_subplot(gs[1, model_idx])
        
        # Calculate metrics (treating any "flag" label as positive prediction)
        def get_metrics(detector_col):
            preds = df_model[detector_col].str.startswith("flag", na=False).fillna(False).astype(int)
            if "gold_label" in df_model.columns:
                gold = df_model["gold_label"].astype(int)
            else:
                # Try to get from raw data
                gold = pd.Series([0] * len(df_model))  # Default if not available
            tp = ((preds == 1) & (gold == 1)).sum()
            fp = ((preds == 1) & (gold == 0)).sum()
            tn = ((preds == 0) & (gold == 0)).sum()
            fn = ((preds == 0) & (gold == 1)).sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            return {"precision": precision, "recall": recall, "f1": f1}
        
        sc_metrics = get_metrics("self_consistency_label")
        temp_metrics = get_metrics("temperature_sensitivity_label")
        
        metrics = ["Precision", "Recall", "F1"]
        sc_values = [sc_metrics["precision"], sc_metrics["recall"], sc_metrics["f1"]]
        temp_values = [temp_metrics["precision"], temp_metrics["recall"], temp_metrics["f1"]]
        
        x = np.arange(len(metrics))
        width = 0.35
        ax2.bar(x - width/2, sc_values, width, label="Self-Consistency", color="#3498db", alpha=0.8)
        ax2.bar(x + width/2, temp_values, width, label="Temperature Sensitivity", color="#e74c3c", alpha=0.8)
        ax2.set_ylabel("Score", fontsize=10)
        ax2.set_title(f"Performance Comparison{title_suffix}", fontsize=11, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics)
        ax2.legend()
        ax2.set_ylim([0, 1.1])
        ax2.grid(True, alpha=0.3, axis="y")
        
        # Add value labels
        for i, (sc_val, temp_val) in enumerate(zip(sc_values, temp_values)):
            ax2.text(i - width/2, sc_val + 0.02, f'{sc_val:.3f}', ha='center', va='bottom', fontsize=8)
            ax2.text(i + width/2, temp_val + 0.02, f'{temp_val:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 3. Computational cost comparison
        ax3 = fig.add_subplot(gs[2, model_idx])
        
        # Use most common values from the data
        sc_most_common = int(df_model["sc_samples"].mode()[0]) if len(df_model["sc_samples"].mode()) > 0 else 4
        temp_most_common = int(df_model["temp_samples"].mode()[0]) if len(df_model["temp_samples"].mode()) > 0 else 6
        
        # Show both actual and potential (if configured differently)
        detectors = ["Self-Consistency\n(actual)", "Temperature\nSensitivity\n(actual)", 
                    "Temperature\nSensitivity\n(potential*)"]
        samples = [sc_most_common, temp_most_common, 4]  # Could use 4 temps × 1 sample = 4
        
        bars = ax3.bar(detectors, samples, color=["#3498db", "#e74c3c", "#27ae60"], alpha=0.8)
        ax3.set_ylabel("Number of Model Samples", fontsize=10)
        ax3.set_title(f"Computational Cost Comparison{title_suffix}", fontsize=11, fontweight="bold")
        ax3.grid(True, alpha=0.3, axis="y")
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10, fontweight="bold")
        
        # Add note about potential efficiency
        ax3.text(0.5, -0.15, "*Could use 4 temperatures × 1 sample = 4 total samples",
                transform=ax3.transAxes, fontsize=8, style='italic', ha='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    plt.suptitle("Temperature Sensitivity vs Self-Consistency: Performance & Efficiency", 
                fontsize=14, fontweight="bold", y=0.995)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_temp_advantages(df: pd.DataFrame, output_path: Path) -> None:
    """Create visualizations highlighting temperature sensitivity advantages."""
    has_model_col = "model" in df.columns and df["model"].nunique() > 1
    models = sorted(df["model"].unique()) if has_model_col else [None]
    
    # Filter to parametric knowledge questions
    df_filtered = df[(df["context"].isna()) | (df["context"].astype(str).str.strip() == "")]
    
    # Load raw data for details
    run_path = find_latest_run(RUNS_DIR)
    raw_records = []
    with run_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_records.append(json.loads(line))
    raw_df = pd.DataFrame(raw_records)
    
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, len(models), hspace=0.35, wspace=0.3)
    
    for model_idx, model in enumerate(models):
        if model:
            df_model = df_filtered[df_filtered["model"] == model]
            title_suffix = f" - {model}"
        else:
            df_model = df_filtered
            title_suffix = ""
        
        if len(df_model) == 0:
            continue
        
        # 1. Score distribution comparison (showing better separation)
        ax1 = fig.add_subplot(gs[0, model_idx])
        
        sc_scores = df_model["self_consistency_score"].fillna(0).values
        temp_scores = df_model["temperature_sensitivity_score"].fillna(0).values
        
        # Get gold labels if available
        has_gold = "gold_label" in df_model.columns
        if has_gold:
            gold = df_model["gold_label"].astype(int).values
            hallucinated_mask = gold == 1
            correct_mask = gold == 0
        else:
            hallucinated_mask = np.array([False] * len(df_model))
            correct_mask = np.array([True] * len(df_model))
        
        # Plot distributions
        if correct_mask.sum() > 0:
            ax1.hist(sc_scores[correct_mask], bins=20, alpha=0.5, label="Self-Consistency (Correct)", 
                    color="#3498db", density=True)
            ax1.hist(temp_scores[correct_mask], bins=20, alpha=0.5, label="Temp Sensitivity (Correct)", 
                    color="#27ae60", density=True)
        if hallucinated_mask.sum() > 0:
            ax1.hist(sc_scores[hallucinated_mask], bins=20, alpha=0.5, label="Self-Consistency (Hallucinated)", 
                    color="#e74c3c", density=True, hatch="//")
            ax1.hist(temp_scores[hallucinated_mask], bins=20, alpha=0.5, label="Temp Sensitivity (Hallucinated)", 
                    color="#c0392b", density=True, hatch="\\\\")
        
        ax1.set_xlabel("Detector Score", fontsize=10)
        ax1.set_ylabel("Density", fontsize=10)
        ax1.set_title(f"Score Distribution & Separation{title_suffix}", fontsize=11, fontweight="bold")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. Cases where temperature sensitivity is better (catches what SC misses)
        ax2 = fig.add_subplot(gs[1, model_idx])
        
        sc_flags = df_model["self_consistency_label"].str.startswith("flag", na=False).fillna(False)
        temp_flags = df_model["temperature_sensitivity_label"].str.startswith("flag", na=False).fillna(False)
        
        # Count different scenarios
        both_flag = ((sc_flags) & (temp_flags)).sum()
        sc_only = ((sc_flags) & (~temp_flags)).sum()
        temp_only = ((~sc_flags) & (temp_flags)).sum()
        neither = ((~sc_flags) & (~temp_flags)).sum()
        
        # Focus on where temp catches what SC misses
        categories = ["Both Flag", "SC Only", "Temp Only\n(Advantage)", "Neither"]
        counts = [both_flag, sc_only, temp_only, neither]
        colors = ["#95a5a6", "#e74c3c", "#27ae60", "#ecf0f1"]
        
        bars = ax2.bar(categories, counts, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)
        ax2.set_ylabel("Number of Questions", fontsize=10)
        ax2.set_title(f"Detection Coverage Comparison{title_suffix}", fontsize=11, fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=9, fontweight="bold")
        
        # Highlight the advantage
        if temp_only > 0:
            ax2.text(2, temp_only + max(counts) * 0.05, 
                    f'Temp catches\n{temp_only} that\nSC misses!',
                    ha='center', fontsize=9, fontweight="bold",
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        # 3. Per-category performance advantage
        ax3 = fig.add_subplot(gs[2, model_idx])
        
        if "category" in df_model.columns:
            categories_list = df_model["category"].unique()
            sc_precisions = []
            temp_precisions = []
            sc_recalls = []
            temp_recalls = []
            cat_names = []
            
            for cat in categories_list:
                cat_df = df_model[df_model["category"] == cat]
                if len(cat_df) < 2:
                    continue
                
                def calc_metrics(detector_col):
                    preds = cat_df[detector_col].str.startswith("flag", na=False).fillna(False).astype(int)
                    if has_gold:
                        gold_cat = cat_df["gold_label"].astype(int)
                        tp = ((preds == 1) & (gold_cat == 1)).sum()
                        fp = ((preds == 1) & (gold_cat == 0)).sum()
                        fn = ((preds == 0) & (gold_cat == 1)).sum()
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        return precision, recall
                    return 0, 0
                
                sc_prec, sc_rec = calc_metrics("self_consistency_label")
                temp_prec, temp_rec = calc_metrics("temperature_sensitivity_label")
                
                sc_precisions.append(sc_prec)
                temp_precisions.append(temp_prec)
                sc_recalls.append(sc_rec)
                temp_recalls.append(temp_rec)
                cat_names.append(cat.replace("_", " ").title())
            
            if cat_names:
                x = np.arange(len(cat_names))
                width = 0.35
                
                # Show F1 advantage
                sc_f1 = [2 * p * r / (p + r) if (p + r) > 0 else 0 for p, r in zip(sc_precisions, sc_recalls)]
                temp_f1 = [2 * p * r / (p + r) if (p + r) > 0 else 0 for p, r in zip(temp_precisions, temp_recalls)]
                
                bars1 = ax3.bar(x - width/2, sc_f1, width, label="Self-Consistency F1", 
                               color="#3498db", alpha=0.8)
                bars2 = ax3.bar(x + width/2, temp_f1, width, label="Temperature Sensitivity F1", 
                               color="#27ae60", alpha=0.8)
                
                ax3.set_ylabel("F1 Score", fontsize=10)
                ax3.set_title(f"Per-Category F1 Comparison{title_suffix}", fontsize=11, fontweight="bold")
                ax3.set_xticks(x)
                ax3.set_xticklabels(cat_names, rotation=45, ha="right", fontsize=8)
                ax3.legend()
                ax3.set_ylim([0, 1.1])
                ax3.grid(True, alpha=0.3, axis="y")
                
                # Highlight where temp is better
                for i, (sc, temp) in enumerate(zip(sc_f1, temp_f1)):
                    if temp > sc:
                        ax3.text(i, max(sc, temp) + 0.05, "✓", ha='center', fontsize=12, 
                                color="green", fontweight="bold")
        else:
            ax3.text(0.5, 0.5, "Category data not available", 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=10)
            ax3.set_title(f"Per-Category Comparison{title_suffix}", fontsize=11, fontweight="bold")
        
        # 4. Computational efficiency timeline
        ax4 = fig.add_subplot(gs[3, model_idx])
        
        # Simulate efficiency over many questions
        num_questions = 1000
        sc_samples = 4
        temp_samples_current = 6
        temp_samples_optimized = 4
        
        questions = np.arange(1, num_questions + 1)
        sc_total = questions * sc_samples
        temp_current_total = questions * temp_samples_current
        temp_optimized_total = questions * temp_samples_optimized
        
        ax4.plot(questions, sc_total, label=f"Self-Consistency ({sc_samples} samples/q)", 
                color="#3498db", linewidth=2)
        ax4.plot(questions, temp_current_total, label=f"Temp Sensitivity Current ({temp_samples_current} samples/q)", 
                color="#e74c3c", linewidth=2, linestyle="--")
        ax4.plot(questions, temp_optimized_total, label=f"Temp Sensitivity Optimized ({temp_samples_optimized} samples/q)", 
                color="#27ae60", linewidth=2, linestyle="-.")
        
        ax4.set_xlabel("Number of Questions", fontsize=10)
        ax4.set_ylabel("Total Model Samples Required", fontsize=10)
        ax4.set_title(f"Scalability & Efficiency{title_suffix}", fontsize=11, fontweight="bold")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add annotation showing savings
        ax4.annotate(f'At 1000 questions:\nOptimized Temp saves\n{sc_total[-1] - temp_optimized_total[-1]:,} samples\n({(1 - temp_samples_optimized/sc_samples)*100:.0f}% reduction)',
                    xy=(1000, temp_optimized_total[-1]), xytext=(600, temp_current_total[-1] + 5000),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2),
                    fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
                    ha='left')
    
    plt.suptitle("Temperature Sensitivity Advantages: Better Separation, Coverage, and Efficiency", 
                fontsize=15, fontweight="bold", y=0.998)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_temp_detailed_analysis(df: pd.DataFrame, output_path: Path) -> None:
    """Detailed analysis showing temperature sensitivity's promise."""
    has_model_col = "model" in df.columns and df["model"].nunique() > 1
    models = sorted(df["model"].unique()) if has_model_col else [None]
    
    df_filtered = df[(df["context"].isna()) | (df["context"].astype(str).str.strip() == "")]
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, len(models), hspace=0.3, wspace=0.3)
    
    for model_idx, model in enumerate(models):
        if model:
            df_model = df_filtered[df_filtered["model"] == model]
            title_suffix = f" - {model}"
        else:
            df_model = df_filtered
            title_suffix = ""
        
        if len(df_model) == 0:
            continue
        
        # 1. Precision-Recall curves comparison
        ax1 = fig.add_subplot(gs[0, model_idx])
        
        sc_scores = df_model["self_consistency_score"].fillna(0).values
        temp_scores = df_model["temperature_sensitivity_score"].fillna(0).values
        
        has_gold = "gold_label" in df_model.columns
        if has_gold:
            gold = df_model["gold_label"].astype(int).values
            
            # Calculate precision-recall at different thresholds
            thresholds = np.linspace(0, 1, 100)
            sc_precisions = []
            sc_recalls = []
            temp_precisions = []
            temp_recalls = []
            
            for thresh in thresholds:
                sc_preds = (sc_scores >= thresh).astype(int)
                temp_preds = (temp_scores >= thresh).astype(int)
                
                # SC metrics
                tp = ((sc_preds == 1) & (gold == 1)).sum()
                fp = ((sc_preds == 1) & (gold == 0)).sum()
                fn = ((sc_preds == 0) & (gold == 1)).sum()
                sc_prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
                sc_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                sc_precisions.append(sc_prec)
                sc_recalls.append(sc_rec)
                
                # Temp metrics
                tp = ((temp_preds == 1) & (gold == 1)).sum()
                fp = ((temp_preds == 1) & (gold == 0)).sum()
                fn = ((temp_preds == 0) & (gold == 1)).sum()
                temp_prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
                temp_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                temp_precisions.append(temp_prec)
                temp_recalls.append(temp_rec)
            
            ax1.plot(sc_recalls, sc_precisions, label="Self-Consistency", 
                    color="#3498db", linewidth=2, alpha=0.7)
            ax1.plot(temp_recalls, temp_precisions, label="Temperature Sensitivity", 
                    color="#27ae60", linewidth=2, alpha=0.7)
            ax1.set_xlabel("Recall", fontsize=10)
            ax1.set_ylabel("Precision", fontsize=10)
            ax1.set_title(f"Precision-Recall Curves{title_suffix}", fontsize=11, fontweight="bold")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim([0, 1.05])
            ax1.set_ylim([0, 1.05])
        else:
            ax1.text(0.5, 0.5, "Gold labels not available\nfor PR curve", 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=10)
            ax1.set_title(f"Precision-Recall Curves{title_suffix}", fontsize=11, fontweight="bold")
        
        # 2. Score separation quality (how well scores separate hallucinations)
        ax2 = fig.add_subplot(gs[1, model_idx])
        
        if has_gold:
            gold = df_model["gold_label"].astype(int).values
            hallucinated_mask = gold == 1
            correct_mask = gold == 0
            
            # Box plots showing separation
            data_to_plot = [
                sc_scores[correct_mask],
                sc_scores[hallucinated_mask],
                temp_scores[correct_mask],
                temp_scores[hallucinated_mask]
            ]
            labels = ["SC\n(Correct)", "SC\n(Halluc)", "Temp\n(Correct)", "Temp\n(Halluc)"]
            colors_box = ["#3498db", "#e74c3c", "#27ae60", "#c0392b"]
            
            bp = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True, 
                           widths=0.6, showmeans=True)
            for patch, color in zip(bp['boxes'], colors_box):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Calculate separation metrics
            sc_separation = sc_scores[hallucinated_mask].mean() - sc_scores[correct_mask].mean() if (hallucinated_mask.sum() > 0 and correct_mask.sum() > 0) else 0
            temp_separation = temp_scores[hallucinated_mask].mean() - temp_scores[correct_mask].mean() if (hallucinated_mask.sum() > 0 and correct_mask.sum() > 0) else 0
            
            ax2.set_ylabel("Detector Score", fontsize=10)
            ax2.set_title(f"Score Separation Quality{title_suffix}\n"
                         f"Separation: SC={sc_separation:.3f}, Temp={temp_separation:.3f} "
                         f"(higher is better)", fontsize=11, fontweight="bold")
            ax2.grid(True, alpha=0.3, axis="y")
            
            # Highlight better separation
            if temp_separation > sc_separation:
                ax2.text(0.5, 0.95, "✓ Temp has better separation!", 
                        transform=ax2.transAxes, ha='center', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
                        fontweight="bold")
        else:
            ax2.text(0.5, 0.5, "Gold labels not available", 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=10)
            ax2.set_title(f"Score Separation{title_suffix}", fontsize=11, fontweight="bold")
    
    plt.suptitle("Detailed Analysis: Why Temperature Sensitivity Shows More Promise", 
                fontsize=15, fontweight="bold", y=0.998)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize QA run logs.")
    parser.add_argument(
        "--run",
        type=Path,
        default=None,
        help="Path to a specific run JSONL file. Defaults to latest in runs/.",
    )
    args = parser.parse_args()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    run_path = args.run or find_latest_run(RUNS_DIR)

    df = load_run(run_path)
    df = unpack_detectors(df)
    df = df.reset_index(drop=True)
    # Use actual qid if available, otherwise use index
    if "qid" not in df.columns:
        df["qid"] = df.index + 1
    else:
        # Ensure qid is string for consistent display
        df["qid"] = df["qid"].astype(str)

    stem = run_path.stem
    counts_path = PLOTS_DIR / f"{stem}_detector_counts.png"
    scatter_path = PLOTS_DIR / f"{stem}_overlap_vs_consistency.png"
    bar_path = PLOTS_DIR / f"{stem}_flagged_questions.png"
    heatmap_path = PLOTS_DIR / f"{stem}_flag_matrix.png"
    map_path = PLOTS_DIR / f"{stem}_question_index.csv"
    samples_path = PLOTS_DIR / f"{stem}_flagged_samples.png"
    samples_csv_path = PLOTS_DIR / f"{stem}_flagged_samples.csv"
    temp_vs_sc_path = PLOTS_DIR / f"{stem}_temp_vs_self_consistency.png"
    temp_advantages_path = PLOTS_DIR / f"{stem}_temp_advantages.png"
    temp_detailed_path = PLOTS_DIR / f"{stem}_temp_detailed_analysis.png"

    plot_detector_counts(df, counts_path)
    plot_overlap_vs_consistency(df, scatter_path)
    plot_flagged_question_bars(df, bar_path)
    plot_flag_matrix(df, heatmap_path)
    df[["qid", "question"]].to_csv(map_path, index=False)
    plot_flagged_responses(df, samples_path)
    plot_temp_vs_self_consistency(df, temp_vs_sc_path)
    plot_temp_advantages(df, temp_advantages_path)
    plot_temp_detailed_analysis(df, temp_detailed_path)

    df[df["self_consistency_label"] == "flag"][
        ["qid", "question", "answer", "self_consistency_unique"]
    ].to_csv(samples_csv_path, index=False)

    print(
        "Created outputs:\n"
        f"- {counts_path}\n"
        f"- {scatter_path}\n"
        f"- {bar_path}\n"
        f"- {heatmap_path}\n"
        f"- {map_path}\n"
        f"- {samples_path}\n"
        f"- {samples_csv_path}\n"
        f"- {temp_vs_sc_path}\n"
        f"- {temp_advantages_path}\n"
        f"- {temp_detailed_path}"
    )


if __name__ == "__main__":
    main()

