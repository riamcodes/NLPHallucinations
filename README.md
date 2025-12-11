# NLP5 Hallucination Detection Baseline

This project explores hallucination detection for generative QA models. It currently focuses on the `google/flan-t5-base` model with optional support for llama.cpp backends (Mistral, Llama-3).

## Quick Start

source venv/bin/activate
python -m pip install -r requirements.txt
python main.py            # run QA + detectors
python visualize_runs.py  # generate plots for the latest run

Outputs are appended to `runs/run-<timestamp>.jsonl` and visualizations are saved in `runs/plots/`.

Gold labels for evaluation live in:

- data/question_set_v1.json – questions + metadata
- data/gold_labels.csv – per-question hallucination labels created by Person A

## Running on a fresh clone
1. Clone the repo and create/activate a virtual environment.

git clone <repo-url>
cd NLP5
python3 -m venv venv
source venv/bin/activate

2. Install dependencies.

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

3. Ensure you have a CUDA/MPS-capable device if desired; FLAN runs on CPU but is faster with GPU/MPS.

4. Optional: set environment variables if you have llama.cpp GGUF weights.

export MISTRAL_GGUF_PATH=/path/to/mistral.gguf
export LLAMA3_GGUF_PATH=/path/to/llama3.gguf

5. Run python main.py to produce answers and detector scores.

6. Run python visualize_runs.py to generate plots summarizing the latest run.

## Updating questions/contexts

The main question set is stored in data/question_set_v1.json, but you can still tweak or extend it:

- Edit or regenerate the JSON used by load_question_set() in questions.py.
- Keep contexts including the ground-truth facts so detectors can judge grounding.
- Re-run python main.py and python visualize_runs.py after making changes.

---

# Hallucination Baseline Overview

This document summarizes the current experiment pipeline so teammates can reproduce and interpret the results.

## Goals
- Stress-test a generative QA model with semi-structured context that may omit or contradict facts.
- Capture signals that indicate potential hallucination: whether the answer is grounded in context and whether the model repeats itself under stochastic sampling.
- Persist outputs and render visual summaries for analysis and presentations.

## Model Selection
- Primary model: google/flan-t5-base
  - Instruction-tuned encoder–decoder that runs comfortably on Apple silicon.
  - Generates free-form answers, so it can hallucinate when context is ambiguous.
  - Open-source weights available via Hugging Face; no external API required.

- Why not DistilBERT/BERT?
  - Extractive QA models only copy spans from context, so they rarely fabricate novel facts.
  - They are useful as baselines but cannot showcase the generative hallucinations we want to study.
  - Generative models (FLAN, Mistral, Llama) can follow context yet still invent details, which is the behaviour the cited papers target.

The code supports additional llama.cpp models when you set MISTRAL_GGUF_PATH and LLAMA3_GGUF_PATH, but the current exploration focuses on FLAN for simplicity.

## Running Experiments

1. Activate the virtual environment and install requirements:

source venv/bin/activate
python -m pip install -r requirements.txt

2. Execute the QA run:

python main.py

- Answers + detector labels print to stdout.
- Results are appended to runs/run-<timestamp>.jsonl.

3. Generate visuals:

python visualize_runs.py

Outputs land in runs/plots/, including PNGs and a CSV mapping question indices to text.

---

## Detectors

Five detectors currently run on every result: three baselines and two experimental detectors added in this project. Each one targets a different hallucination signal: grounding, stability, context dependence, temperature sensitivity, and semantic similarity.

### Context Overlap (detectors/context_overlap.py)

- Tokenizes answer and context, counts overlapping words.
- Computes: support_ratio = overlap_tokens / answer_tokens.
- Threshold: 0.25.
  - ratio >= 0.25 → support
  - ratio < 0.25 → flag
  - If either side lacks tokens → insufficient
- High support suggests grounding; low support indicates novel unsupported content.

### Self Consistency (detectors/self_consistency.py)

- Draws 5 stochastic samples (temperature 0.6).
- Computes diversity = (# unique normalized answers) / 5.
- Threshold: 0.4.
  - > 0.4 → flag
  - <= 0.4 → consistent
  - If sampling unsupported → unsupported

### Context Ablation (detectors/context_ablation.py)

- Finds the most supporting sentence in context.
- Removes it to create an ablated variant.
- Regenerates answer; computes sensitivity = 1 − overlap_ratio(original, ablated).
- Thresholds:
  - support_threshold = 0.2
  - sensitivity_threshold = 0.3

Labels:
- flag_context_ignored
- context_sensitive
- insufficient

This detector checks whether the model actually relies on the given context.

### Temperature Sensitivity (detectors/temperature_sensitivity.py)

- Samples answers at T = 0.3, 0.7, 1.0.
- Computes drift = 1 − overlap_ratio(baseline, sample).
- Aggregates into mean_drift and max_drift.

Flags when:
- mean_drift ≥ 0.35 OR
- max_drift ≥ 0.60.

Labels:
- flag_temp_sensitive
- temperature_stable
- insufficient

Interpretation: high-temperature divergence suggests instability and hallucination likelihood.

### Semantic Overlap (detectors/semantic_overlap.py)

- Encodes context and answer with all-MiniLM-L6-v2.
- Computes cosine similarity.
- Flags when similarity < 0.60.

Labels:
- support
- flag
- insufficient/unsupported

Semantic overlap detects meaning drift beyond token matching.

### Shared Labeling Convention

Any label starting with "flag" is treated as "hallucination" by evaluate_detectors.py.

---

## Detector Evaluation & Ensembles (analysis/evaluate_detectors.py)

After generating a run via python main.py, evaluate detectors via:

python analysis/evaluate_detectors.py --save-csv

This prints metrics for:
- Individual detectors
- Ensembles: ens_any, ens_majority, ens_all, ens_weighted

And saves a detailed CSV to results/detector_eval_<run-stem>.csv.

### Ensemble Methods

- ens_any → flag if any detector flags.
- ens_majority → flag if ≥ 2 detectors flag.
- ens_all → flag only if all detectors flag.
- ens_weighted → weighted sum across detectors using:

context_overlap: 0.25
self_consistency: 0.25
context_ablation: 0.20
temperature_sensitivity: 0.15
semantic_overlap: 0.15

Predict hallucination when score >= 0.5.

---

## Visualizations (runs/plots/)

- *_detector_counts.png – histogram of detector label frequencies.
- *_overlap_vs_consistency.png – scatter of lexical support vs consistency.
- *_flagged_questions.png – ranking of questions by flags.
- *_flag_matrix.png – heatmap of detectors × questions.
- *_flagged_samples.* – sampled disagreement visualizations.
- *_question_index.csv – lookup table mapping QIDs to text.

---

## Extending the Pipeline

- Add new detectors by writing files under detectors/ and editing main.py logging.
- Tune thresholds and sampling strategies.
- Compare FLAN to Llama/Mistral by providing GGUF model paths.
- Add larger datasets by adapting questions.py.

---

## References

- Huang et al. (2025) — hallucination taxonomy.
- Kryscinski et al. (2020) — factual consistency metrics.
- Manakul et al. (2023) — SelfCheckGPT.

---

## Appendix: Simple Question Answering with DistilBERT and GPT-2

(This demo is preserved for reference; not used in the hallucination pipeline.)

Requirements:
- torch
- transformers

Example usage:

python3 -m venv venv
source venv/bin/activate
pip install torch transformers
python main.py

Edit the questions list in main.py to customize behavior. 