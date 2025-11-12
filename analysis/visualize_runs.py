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
    df["context_overlap_score"] = df.apply(
        lambda row: get_detector_value(row, "context_overlap", "score"), axis=1
    )
    df["context_overlap_label"] = df.apply(
        lambda row: get_detector_value(row, "context_overlap", "label"), axis=1
    )
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
    df["flagged"] = df.apply(
        lambda row: any(
            label == "flag"
            for label in (
                row.get("context_overlap_label"),
                row.get("self_consistency_label"),
            )
            if label
        ),
        axis=1,
    )
    return df


def plot_detector_counts(df: pd.DataFrame, output_path: Path) -> None:
    counts = (
        pd.melt(
            df,
            id_vars=["question"],
            value_vars=["context_overlap_label", "self_consistency_label"],
            var_name="detector",
            value_name="label",
        )
        .groupby(["detector", "label"], dropna=False)
        .size()
        .reset_index(name="count")
    )

    counts["detector"] = counts["detector"].str.replace("_label", "", regex=False)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 4))
    sns.barplot(data=counts, x="detector", y="count", hue="label", palette="deep")
    plt.title("Detector label counts")
    plt.ylabel("Count")
    plt.xlabel("Detector")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_overlap_vs_consistency(df: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(7, 6))

    jitter_rng = np.random.default_rng(42)
    df = df.copy()
    df["context_jitter"] = df["context_overlap_score"] + jitter_rng.uniform(-0.02, 0.02, len(df))
    df["consistency_jitter"] = df["self_consistency_score"] + jitter_rng.uniform(-0.02, 0.02, len(df))

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
    df = df.copy()
    df["flag_count"] = df.apply(
        lambda row: sum(
            label == "flag"
            for label in (
                row.get("context_overlap_label"),
                row.get("self_consistency_label"),
            )
            if label
        ),
        axis=1,
    )
    df = df.sort_values("flag_count", ascending=False)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 8))
    sns.barplot(
        data=df,
        y="qid",
        x="flag_count",
        color="#d62728",
        legend=False,
    )
    plt.xlabel("Number of detectors flagging")
    plt.ylabel("Question index")
    plt.title("Flagged detectors per question")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_flag_matrix(df: pd.DataFrame, output_path: Path) -> None:
    matrix = pd.DataFrame(
        {
            "context_overlap": (df["context_overlap_label"] == "flag").astype(int),
            "self_consistency": (df["self_consistency_label"] == "flag").astype(int),
        },
        index=df["qid"],
    )

    cmap = ListedColormap(["#f0f0f0", "#d62728"])
    sns.set_theme(style="white")
    plt.figure(figsize=(6, 6))
    annot_values = matrix.values.astype(int)
    sns.heatmap(
        matrix,
        cmap=cmap,
        linewidths=0.5,
        linecolor="#cccccc",
        cbar=False,
        annot=annot_values,
        fmt="d",
        annot_kws={"fontsize": 9},
        vmin=0,
        vmax=1,
    )
    plt.xlabel("Detector")
    plt.ylabel("Question index")
    plt.title("Detector flags by question (1 = flag)")
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
    df["qid"] = df.index + 1

    stem = run_path.stem
    counts_path = PLOTS_DIR / f"{stem}_detector_counts.png"
    scatter_path = PLOTS_DIR / f"{stem}_overlap_vs_consistency.png"
    bar_path = PLOTS_DIR / f"{stem}_flagged_questions.png"
    heatmap_path = PLOTS_DIR / f"{stem}_flag_matrix.png"
    map_path = PLOTS_DIR / f"{stem}_question_index.csv"
    samples_path = PLOTS_DIR / f"{stem}_flagged_samples.png"
    samples_csv_path = PLOTS_DIR / f"{stem}_flagged_samples.csv"

    plot_detector_counts(df, counts_path)
    plot_overlap_vs_consistency(df, scatter_path)
    plot_flagged_question_bars(df, bar_path)
    plot_flag_matrix(df, heatmap_path)
    df[["qid", "question"]].to_csv(map_path, index=False)
    plot_flagged_responses(df, samples_path)

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
        f"- {samples_csv_path}"
    )


if __name__ == "__main__":
    main()

