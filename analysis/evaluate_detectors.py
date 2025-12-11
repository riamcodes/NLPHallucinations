"""
Evaluate hallucination detectors against gold labels.

- Loads a run JSONL file from runs/run-*.jsonl
- Joins each row on gold_labels.csv via qid
- Treats each detector's "flag*" label as predicting "hallucination"
- Builds simple ensembles (any-flag, majority-flag, all-flag)
- Builds a weighted ensemble (ens_weighted)
- Prints precision / recall / F1 / accuracy overall and by category
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

RUNS_DIR = Path("runs")
DATA_DIR = Path("data")
GOLD_PATH = DATA_DIR / "gold_labels.csv"

# ---------------------------------------------------------------------------
# Detector configuration
# ---------------------------------------------------------------------------

# Detectors we expect to see in run JSONL files
DETECTOR_NAMES = [
    "context_overlap",
    "self_consistency",
    "context_ablation",
    "temperature_sensitivity",
    "semantic_overlap",
]

# Weights for the weighted ensemble (sum does not have to be 1.0)
WEIGHTED_ENSEMBLE_WEIGHTS: Dict[str, float] = {
    "context_overlap": 0.25,
    "self_consistency": 0.25,
    "context_ablation": 0.20,
    "temperature_sensitivity": 0.15,
    "semantic_overlap": 0.15,
}

# Threshold on the weighted score to predict hallucination
WEIGHTED_ENSEMBLE_THRESHOLD: float = 0.5


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def find_latest_run(run_dir: Path) -> Path:
    """Return the most recent JSONL run file."""
    candidates = sorted(run_dir.glob("run-*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No run files found in {run_dir!s}")
    return candidates[-1]


def load_run(path: Path) -> List[dict]:
    """Load a JSONL run file into a list of dicts."""
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"Run file {path} is empty.")
    return records


def load_gold(path: Path) -> Dict[str, dict]:
    """
    Load gold labels as a mapping qid -> record.

    Expects columns:
      qid, category, question, is_hallucination, explanation
    """
    import csv

    gold_by_qid: Dict[str, dict] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row.get("qid")
            if not qid:
                continue
            raw_label = (row.get("is_hallucination") or "").strip().lower()
            # Accept "true"/"false", "1"/"0", "yes"/"no"
            if raw_label in {"true", "1", "yes"}:
                is_hallu = True
            elif raw_label in {"false", "0", "no"}:
                is_hallu = False
            else:
                # If it's something unexpected, skip this row
                continue

            gold_by_qid[qid] = {
                "qid": qid,
                "category": (row.get("category") or "").strip(),
                "question": row.get("question") or "",
                "is_hallucination": is_hallu,
                "explanation": row.get("explanation") or "",
            }
    if not gold_by_qid:
        raise ValueError(f"No usable gold labels found in {path}")
    return gold_by_qid


@dataclass
class ConfusionCounts:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def update(self, gold: bool, pred: bool) -> None:
        if gold and pred:
            self.tp += 1
        elif not gold and pred:
            self.fp += 1
        elif not gold and not pred:
            self.tn += 1
        elif gold and not pred:
            self.fn += 1


@dataclass
class Metrics:
    name: str
    n: int
    accuracy: float
    precision: float
    recall: float
    f1: float


def compute_metrics(name: str, counts: ConfusionCounts) -> Metrics:
    tp, fp, tn, fn = counts.tp, counts.fp, counts.tn, counts.fn
    n = tp + fp + tn + fn

    accuracy = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return Metrics(
        name=name,
        n=n,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
    )


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def detector_pred(row: dict, det_name: str) -> bool:
    """
    Map detector output -> binary hallucination prediction.

    Convention:
      any label starting with "flag" -> predict hallucination (True)
      anything else                  -> predict non-hallucination (False)

    This allows labels like "flag_context_ignored" or
    "flag_temp_sensitive" to count as positives.
    """
    dets = row.get("detections") or {}
    payload = dets.get(det_name) or {}
    label = (payload.get("label") or "").lower()
    return label.startswith("flag")


def build_predictions(rows: Iterable[dict]) -> List[dict]:
    """
    For each row, build a dict with:
      qid, category, gold,
      detector predictions,
      ensemble predictions (any / majority / all / weighted)
    """
    # Gold labels
    gold = load_gold(GOLD_PATH)

    examples: List[dict] = []

    skipped_no_qid = 0
    skipped_no_gold = 0

    for row in rows:
        qid = row.get("qid")
        if not qid:
            skipped_no_qid += 1
            continue

        gold_row = gold.get(str(qid))
        if not gold_row:
            skipped_no_gold += 1
            continue

        gold_label = bool(gold_row["is_hallucination"])

        # Individual detector predictions
        det_preds = {name: detector_pred(row, name) for name in DETECTOR_NAMES}
        num_flags = sum(det_preds.values())

        # Simple ensembles
        ensemble_any = num_flags >= 1
        ensemble_majority = num_flags >= 2
        ensemble_all = num_flags == len(DETECTOR_NAMES)

        # Weighted ensemble
        weighted_score = 0.0
        for det_name, pred in det_preds.items():
            if not pred:
                continue
            weight = WEIGHTED_ENSEMBLE_WEIGHTS.get(det_name, 0.0)
            weighted_score += weight
        ensemble_weighted = weighted_score >= WEIGHTED_ENSEMBLE_THRESHOLD

        examples.append(
            {
                "qid": str(qid),
                "category": gold_row["category"],
                "gold": gold_label,
                **{f"det_{k}": v for k, v in det_preds.items()},
                "ens_any": ensemble_any,
                "ens_majority": ensemble_majority,
                "ens_all": ensemble_all,
                "ens_weighted_score": weighted_score,
                "ens_weighted": ensemble_weighted,
            }
        )

    if skipped_no_qid or skipped_no_gold:
        print(
            f"NOTE: skipped {skipped_no_qid} rows without qid "
            f"and {skipped_no_gold} rows without matching gold label."
        )

    return examples


def aggregate_metrics(
    examples: List[dict],
) -> Tuple[Dict[str, Metrics], Dict[str, Dict[str, Metrics]]]:
    """
    Compute metrics overall and per category.

    Returns:
      overall_metrics: name -> Metrics
      per_category_metrics: category -> (name -> Metrics)
    """
    overall_counts: Dict[str, ConfusionCounts] = defaultdict(ConfusionCounts)
    per_category_counts: Dict[str, Dict[str, ConfusionCounts]] = defaultdict(
        lambda: defaultdict(ConfusionCounts)
    )

    model_names = (
        [f"det_{n}" for n in DETECTOR_NAMES]
        + ["ens_any", "ens_majority", "ens_all", "ens_weighted"]
    )

    for ex in examples:
        gold = ex["gold"]
        cat = ex["category"] or "uncategorised"

        for name in model_names:
            pred = bool(ex[name])
            overall_counts[name].update(gold, pred)
            per_category_counts[cat][name].update(gold, pred)

    overall_metrics = {
        name: compute_metrics(name, counts)
        for name, counts in overall_counts.items()
    }

    per_category_metrics: Dict[str, Dict[str, Metrics]] = {}
    for cat, counts_by_name in per_category_counts.items():
        per_category_metrics[cat] = {
            name: compute_metrics(name, counts)
            for name, counts in counts_by_name.items()
        }

    return overall_metrics, per_category_metrics


def print_metrics_table(title: str, metrics_by_name: Dict[str, Metrics]) -> None:
    print()
    print(title)
    print("-" * len(title))
    header = f"{'name':20s} {'N':>5s} {'acc':>7s} {'prec':>7s} {'recall':>7s} {'F1':>7s}"
    print(header)
    print("-" * len(header))
    for name in sorted(metrics_by_name.keys()):
        m = metrics_by_name[name]
        print(
            f"{m.name:20s} "
            f"{m.n:5d} "
            f"{m.accuracy:7.3f} "
            f"{m.precision:7.3f} "
            f"{m.recall:7.3f} "
            f"{m.f1:7.3f}"
        )
    print()


def save_metrics_csv(out_path: Path, examples: List[dict]) -> None:
    """Optional: save per-example predictions for debugging."""
    import csv

    if not examples:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(examples[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(examples)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate hallucination detectors against gold labels."
    )
    parser.add_argument(
        "--run",
        type=Path,
        default=None,
        help="Path to a specific run JSONL file. Defaults to latest run in runs/.",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="If set, save per-example predictions to results/detector_eval_<stem>.csv",
    )
    args = parser.parse_args()

    GOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not GOLD_PATH.exists():
        raise FileNotFoundError(f"Gold label file not found at {GOLD_PATH}")

    run_path = args.run or find_latest_run(RUNS_DIR)
    print(f"Using run file: {run_path}")

    rows = load_run(run_path)
    examples = build_predictions(rows)
    if not examples:
        print("No examples with both qid and gold label. Nothing to evaluate.")
        return

    overall_metrics, per_category_metrics = aggregate_metrics(examples)

    # Overall
    print_metrics_table("Overall metrics", overall_metrics)

    # Per category
    for cat, metrics in sorted(per_category_metrics.items()):
        title = f"Category: {cat}"
        print_metrics_table(title, metrics)

    # Optional CSV dump
    if args.save_csv:
        out_dir = Path("results")
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = run_path.stem
        out_path = out_dir / f"detector_eval_{stem}.csv"
        save_metrics_csv(out_path, examples)
        print(f"Saved per-example predictions to {out_path}")


if __name__ == "__main__":
    main()
