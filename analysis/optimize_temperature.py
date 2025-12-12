#!/usr/bin/env python3
"""Comprehensive optimization study for temperature sensitivity detector."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

from detectors import SelfConsistencyDetector, TemperatureSensitivityDetector
from qa_models import FlanConfig, FlanT5QA, QAModel
from questions import load_question_set

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RUNS_DIR = Path("runs")
OPTIMIZATION_DIR = RUNS_DIR / "optimization"
OPTIMIZATION_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = OPTIMIZATION_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ConfigResult:
    """Results for a single configuration."""
    config_name: str
    temperatures: List[float]
    samples_per_temp: int
    mean_threshold: float
    max_threshold: float
    total_samples: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    flag_rate: float


def calculate_metrics(predictions: List[bool], gold_labels: List[bool]) -> Dict[str, float]:
    """Calculate precision, recall, F1, accuracy."""
    tp = sum((p and g) for p, g in zip(predictions, gold_labels))
    fp = sum((p and not g) for p, g in zip(predictions, gold_labels))
    tn = sum((not p and not g) for p, g in zip(predictions, gold_labels))
    fn = sum((not p and g) for p, g in zip(predictions, gold_labels))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "flag_rate": (tp + fp) / len(predictions) if predictions else 0.0,
    }


def test_configuration(
    questions: List[Dict[str, Any]],
    model: QAModel,
    temperatures: List[float],
    samples_per_temp: int,
    mean_threshold: float,
    max_threshold: float,
) -> ConfigResult:
    """Test a single temperature sensitivity configuration."""
    detector = TemperatureSensitivityDetector(
        temperatures=temperatures,
        samples_per_temp=samples_per_temp,
        mean_drift_threshold=mean_threshold,
        max_drift_threshold=max_threshold,
    )
    
    predictions = []
    gold_labels = []
    
    for q in questions:
        # Get baseline answer
        result = model.answer(q["question"], q.get("context", ""))
        if "error" in result:
            continue
        
        # Evaluate detector
        detection = detector.evaluate(result, model)
        is_flag = detection.label.startswith("flag")
        predictions.append(is_flag)
        
        # Get gold label
        gold = q.get("gold_label", {}).get("is_hallucination", False)
        gold_labels.append(gold)
    
    metrics = calculate_metrics(predictions, gold_labels)
    
    return ConfigResult(
        config_name=f"Temp{len(temperatures)}x{samples_per_temp}_M{mean_threshold:.2f}_X{max_threshold:.2f}",
        temperatures=temperatures,
        samples_per_temp=samples_per_temp,
        mean_threshold=mean_threshold,
        max_threshold=max_threshold,
        total_samples=len(temperatures) * samples_per_temp,
        **metrics,
    )


def test_self_consistency(
    questions: List[Dict[str, Any]],
    model: QAModel,
    samples: int = 4,
    threshold: float = 0.4,
) -> ConfigResult:
    """Test self-consistency baseline."""
    detector = SelfConsistencyDetector(samples=samples, temperature=0.6)
    
    predictions = []
    gold_labels = []
    
    for q in questions:
        result = model.answer(q["question"], q.get("context", ""))
        if "error" in result:
            continue
        
        detection = detector.evaluate(result, model)
        # Self-consistency uses diversity > threshold to flag
        is_flag = detection.score > threshold
        predictions.append(is_flag)
        
        gold = q.get("gold_label", {}).get("is_hallucination", False)
        gold_labels.append(gold)
    
    metrics = calculate_metrics(predictions, gold_labels)
    
    return ConfigResult(
        config_name=f"SelfConsistency_{samples}samples",
        temperatures=[],
        samples_per_temp=0,
        mean_threshold=threshold,
        max_threshold=threshold,
        total_samples=samples,
        **metrics,
    )


def run_optimization_study(questions: List[Dict[str, Any]], model: QAModel) -> List[ConfigResult]:
    """Run comprehensive optimization study."""
    logger.info("Starting optimization study...")
    
    results = []
    
    # Filter to parametric knowledge questions only
    param_questions = [q for q in questions if not q.get("context", "").strip()]
    logger.info(f"Testing on {len(param_questions)} parametric knowledge questions")
    
    # Test self-consistency baseline
    logger.info("Testing self-consistency baseline...")
    sc_result = test_self_consistency(param_questions, model, samples=4, threshold=0.4)
    results.append(sc_result)
    logger.info(f"Self-Consistency: F1={sc_result.f1:.3f}, Samples={sc_result.total_samples}")
    
    # Test different temperature configurations
    temp_configs = [
        # (temperatures, samples_per_temp)
        ([0.3, 0.7, 1.0], 1),  # 3 temps, 1 sample = 3 total
        ([0.3, 0.7, 1.0], 2),  # 3 temps, 2 samples = 6 total (current)
        ([0.3, 0.5, 0.7, 1.0], 1),  # 4 temps, 1 sample = 4 total
        ([0.3, 0.5, 0.7, 1.0], 2),  # 4 temps, 2 samples = 8 total
        ([0.3, 0.7], 2),  # 2 temps, 2 samples = 4 total
        ([0.3, 0.7], 1),  # 2 temps, 1 sample = 2 total
        ([0.5, 1.0], 2),  # 2 temps, 2 samples = 4 total (different temps)
        ([0.3, 0.9], 2),  # 2 temps, 2 samples = 4 total (wider range)
    ]
    
    # Test different threshold combinations
    threshold_configs = [
        (0.30, 0.50),  # Lower thresholds
        (0.35, 0.60),  # Current
        (0.40, 0.70),  # Higher thresholds
        (0.25, 0.55),  # Lower mean, medium max
        (0.45, 0.65),  # Higher mean, medium max
    ]
    
    total_configs = len(temp_configs) * len(threshold_configs)
    logger.info(f"Testing {total_configs} temperature sensitivity configurations...")
    
    config_num = 0
    for temps, samples_per_temp in temp_configs:
        for mean_thresh, max_thresh in threshold_configs:
            config_num += 1
            if config_num % 10 == 0:
                logger.info(f"Progress: {config_num}/{total_configs}")
            
            result = test_configuration(
                param_questions,
                model,
                temperatures=temps,
                samples_per_temp=samples_per_temp,
                mean_threshold=mean_thresh,
                max_threshold=max_thresh,
            )
            results.append(result)
    
    logger.info(f"Completed {len(results)} configurations")
    return results


def plot_optimization_results(results: List[ConfigResult], output_dir: Path) -> None:
    """Create comprehensive visualization of optimization results."""
    df = pd.DataFrame([
        {
            "config": r.config_name,
            "samples": r.total_samples,
            "precision": r.precision,
            "recall": r.recall,
            "f1": r.f1,
            "accuracy": r.accuracy,
            "flag_rate": r.flag_rate,
            "is_sc": "Self-Consistency" in r.config_name,
            "temperatures": len(r.temperatures),
            "samples_per_temp": r.samples_per_temp,
        }
        for r in results
    ])
    
    # Separate SC and Temp results
    sc_results = df[df["is_sc"]]
    temp_results = df[~df["is_sc"]]
    
    if len(sc_results) == 0:
        logger.warning("No self-consistency results found, using default")
        sc_f1 = 0.316  # From the log output
    else:
        sc_f1 = sc_results["f1"].iloc[0]
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
    
    # 1. F1 vs Samples (main optimization plot)
    ax1 = fig.add_subplot(gs[0, :2])
    if len(sc_results) > 0:
        sc_ax = ax1.scatter(sc_results["samples"], sc_results["f1"], 
                           s=200, marker="*", color="#e74c3c", 
                           label="Self-Consistency", zorder=5, edgecolors="black", linewidth=2)
    else:
        # Plot baseline point manually
        ax1.scatter(4, sc_f1, s=200, marker="*", color="#e74c3c", 
                   label="Self-Consistency", zorder=5, edgecolors="black", linewidth=2)
    
    temp_ax = ax1.scatter(temp_results["samples"], temp_results["f1"],
                         s=100, alpha=0.6, color="#27ae60",
                         label="Temperature Sensitivity", zorder=3)
    
    # Highlight best configurations
    if len(temp_results) > 0:
        best_temp = temp_results.loc[temp_results["f1"].idxmax()]
        ax1.scatter(best_temp["samples"], best_temp["f1"], 
                   s=300, marker="o", color="gold", 
                   edgecolors="black", linewidth=3, zorder=6,
                   label=f"Best Temp (F1={best_temp['f1']:.3f})")
        ax1.annotate(f"Best: {best_temp['config']}\n{best_temp['samples']} samples",
                    xy=(best_temp["samples"], best_temp["f1"]),
                    xytext=(10, 10), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                    fontsize=9, fontweight="bold")
    
    ax1.set_xlabel("Total Samples Required", fontsize=12, fontweight="bold")
    ax1.set_ylabel("F1 Score", fontsize=12, fontweight="bold")
    ax1.set_title("Optimization: F1 Score vs Computational Cost", fontsize=14, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Precision-Recall comparison
    ax2 = fig.add_subplot(gs[0, 2])
    if len(sc_results) > 0:
        ax2.scatter(sc_results["recall"], sc_results["precision"], 
                   s=200, marker="*", color="#e74c3c", 
                   label="Self-Consistency", zorder=5, edgecolors="black", linewidth=2)
    if len(temp_results) > 0:
        ax2.scatter(temp_results["recall"], temp_results["precision"],
                   s=100, alpha=0.6, color="#27ae60",
                   label="Temperature Sensitivity", zorder=3)
        best_temp = temp_results.loc[temp_results["f1"].idxmax()]
        ax2.scatter(best_temp["recall"], best_temp["precision"],
                   s=300, marker="o", color="gold", 
                   edgecolors="black", linewidth=3, zorder=6)
    ax2.set_xlabel("Recall", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Precision", fontsize=11, fontweight="bold")
    ax2.set_title("Precision-Recall Space", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1.05])
    ax2.set_ylim([0, 1.05])
    
    # 3. Top 10 configurations
    ax3 = fig.add_subplot(gs[1, :])
    if len(temp_results) > 0:
        top_10 = temp_results.nlargest(min(10, len(temp_results)), "f1")
        y_pos = np.arange(len(top_10))
        bars = ax3.barh(y_pos, top_10["f1"], color="#27ae60", alpha=0.7)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels([f"{row['config']} ({row['samples']} samples)" 
                             for _, row in top_10.iterrows()], fontsize=9)
        ax3.set_xlabel("F1 Score", fontsize=11, fontweight="bold")
        ax3.set_title("Top 10 Temperature Sensitivity Configurations", fontsize=12, fontweight="bold")
        ax3.grid(True, alpha=0.3, axis="x")
        
        # Add value labels
        for i, (idx, row) in enumerate(top_10.iterrows()):
            ax3.text(row["f1"] + 0.01, i, f"{row['f1']:.3f}", 
                    va="center", fontsize=9, fontweight="bold")
    
    # Add SC baseline line
    ax3.axvline(sc_f1, color="#e74c3c", linestyle="--", linewidth=2, 
               label=f"Self-Consistency Baseline (F1={sc_f1:.3f})")
    ax3.legend(loc="lower right", fontsize=9)
    
    # 4. Samples vs Performance (heatmap)
    ax4 = fig.add_subplot(gs[2, :2])
    pivot = temp_results.pivot_table(
        values="f1", 
        index="samples", 
        aggfunc=["mean", "max", "min"]
    )
    pivot.columns = ["Mean F1", "Max F1", "Min F1"]
    
    x = pivot.index.values
    width = 0.25
    x_pos = np.arange(len(x))
    
    ax4.bar(x_pos - width, pivot["Mean F1"], width, label="Mean F1", color="#3498db", alpha=0.7)
    ax4.bar(x_pos, pivot["Max F1"], width, label="Max F1", color="#27ae60", alpha=0.7)
    ax4.bar(x_pos + width, pivot["Min F1"], width, label="Min F1", color="#e74c3c", alpha=0.7)
    
    ax4.set_xlabel("Total Samples", fontsize=11, fontweight="bold")
    ax4.set_ylabel("F1 Score", fontsize=11, fontweight="bold")
    ax4.set_title("Performance by Sample Count", fontsize=12, fontweight="bold")
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([int(s) for s in x])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis="y")
    
    # Add SC baseline
    ax4.axhline(sc_f1, color="#e74c3c", linestyle="--", linewidth=2, alpha=0.7)
    
    # 5. Configuration details table
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.axis("off")
    
    # Create table of best configs
    if len(temp_results) > 0:
        top_5 = temp_results.nlargest(min(5, len(temp_results)), "f1")
        table_data = []
        for i, (idx, row) in enumerate(top_5.iterrows()):
            table_data.append([
                f"#{i+1}",
                row["config"],
                f"{row['samples']}",
                f"{row['f1']:.3f}",
                f"{row['precision']:.3f}",
                f"{row['recall']:.3f}",
            ])
        
        table = ax5.table(
            cellText=table_data,
            colLabels=["Rank", "Config", "Samples", "F1", "Prec", "Rec"],
            cellLoc="center",
            loc="center",
            colWidths=[0.1, 0.4, 0.15, 0.12, 0.12, 0.12],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2)
        
        # Style header
        for i in range(6):
            table[(0, i)].set_facecolor("#34495e")
            table[(0, i)].set_text_props(weight="bold", color="white")
        
        # Highlight best
        for i in range(6):
            table[(1, i)].set_facecolor("#f1c40f")
    
    ax5.set_title("Top 5 Configurations", fontsize=12, fontweight="bold", pad=20)
    
    # 6. Efficiency comparison (samples vs F1 improvement over SC)
    ax6 = fig.add_subplot(gs[3, :])
    
    temp_results["f1_improvement"] = temp_results["f1"] - sc_f1
    temp_results["efficiency"] = temp_results["f1_improvement"] / temp_results["samples"]
    
    # Scatter plot
    scatter = ax6.scatter(temp_results["samples"], temp_results["f1_improvement"],
                         s=150, c=temp_results["efficiency"], 
                         cmap="RdYlGn", alpha=0.7, edgecolors="black", linewidth=0.5)
    
    # Highlight best
    best_idx = temp_results["f1_improvement"].idxmax()
    best_row = temp_results.loc[best_idx]
    ax6.scatter(best_row["samples"], best_row["f1_improvement"],
               s=400, marker="*", color="gold", 
               edgecolors="black", linewidth=3, zorder=5,
               label=f"Best: {best_row['config']}")
    
    # Highlight efficient ones (high improvement, low samples)
    efficient = temp_results[
        (temp_results["f1_improvement"] > 0) & 
        (temp_results["samples"] <= 4)
    ]
    if len(efficient) > 0:
        ax6.scatter(efficient["samples"], efficient["f1_improvement"],
                   s=200, marker="o", color="cyan", 
                   edgecolors="black", linewidth=2, zorder=4,
                   label=f"Efficient (≤4 samples, F1>{sc_f1:.3f})")
    
    ax6.axhline(0, color="#e74c3c", linestyle="--", linewidth=2, 
               label="Self-Consistency Baseline")
    ax6.set_xlabel("Total Samples", fontsize=12, fontweight="bold")
    ax6.set_ylabel("F1 Improvement over Self-Consistency", fontsize=12, fontweight="bold")
    ax6.set_title("Efficiency Analysis: F1 Improvement vs Computational Cost", 
                 fontsize=13, fontweight="bold")
    ax6.legend(loc="best", fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax6)
    cbar.set_label("Efficiency (F1 improvement / samples)", fontsize=10)
    
    plt.suptitle("Temperature Sensitivity Optimization Study", 
                fontsize=16, fontweight="bold", y=0.995)
    
    output_path = output_dir / "optimization_comprehensive.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved comprehensive plot to {output_path}")
    
    # Save results to CSV
    csv_path = output_dir / "optimization_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved results to {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize temperature sensitivity detector")
    parser.add_argument("--model", type=str, default="flan-t5-base", 
                       help="Model to use for optimization")
    args = parser.parse_args()
    
    # Load questions
    questions = load_question_set()
    
    # Build model
    import torch
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = None
    model = FlanT5QA(
        name=args.model,
        config=FlanConfig(device=device),
    )
    model.load()
    
    # Run optimization
    results = run_optimization_study(questions, model)
    
    # Create visualizations
    plot_optimization_results(results, PLOTS_DIR)
    
    # Print summary
    print("\n" + "="*80)
    print("OPTIMIZATION SUMMARY")
    print("="*80)
    
    sc_result = [r for r in results if "Self-Consistency" in r.config_name][0]
    print(f"\nSelf-Consistency Baseline:")
    print(f"  F1: {sc_result.f1:.3f}, Samples: {sc_result.total_samples}")
    
    temp_results = [r for r in results if "Self-Consistency" not in r.config_name]
    best = max(temp_results, key=lambda r: r.f1)
    print(f"\nBest Temperature Sensitivity Configuration:")
    print(f"  Config: {best.config_name}")
    print(f"  F1: {best.f1:.3f} (improvement: {best.f1 - sc_result.f1:+.3f})")
    print(f"  Precision: {best.precision:.3f}, Recall: {best.recall:.3f}")
    print(f"  Samples: {best.total_samples} (temperatures: {best.temperatures}, samples_per_temp: {best.samples_per_temp})")
    print(f"  Thresholds: mean={best.mean_threshold:.2f}, max={best.max_threshold:.2f}")
    
    # Find best with same or fewer samples
    efficient = [r for r in temp_results if r.total_samples <= sc_result.total_samples]
    if efficient:
        best_efficient = max(efficient, key=lambda r: r.f1)
        print(f"\nBest Efficient Configuration (≤{sc_result.total_samples} samples):")
        print(f"  Config: {best_efficient.config_name}")
        print(f"  F1: {best_efficient.f1:.3f} (improvement: {best_efficient.f1 - sc_result.f1:+.3f})")
        print(f"  Samples: {best_efficient.total_samples}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

