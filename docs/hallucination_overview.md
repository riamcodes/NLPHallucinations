# Hallucination Baseline Overview

This document summarizes the current experiment pipeline so teammates can reproduce and interpret the results.

## Goals
- Stress-test a generative QA model with semi-structured context that may omit or contradict facts.
- Capture signals that indicate potential hallucination: whether the answer is grounded in context and whether the model repeats itself under stochastic sampling.
- Persist outputs and render visual summaries for analysis and presentations.

## Model Selection
- **Primary model:** `google/flan-t5-base`
  - Instruction-tuned encoder–decoder that runs comfortably on Apple silicon.
  - Generates free-form answers, so it can hallucinate when context is ambiguous.
  - Open-source weights available via Hugging Face; no external API required.
- **Why not DistilBERT/BERT?**
  - Extractive QA models only copy spans from the context, so they rarely fabricate novel facts.
  - They are useful as baselines but cannot showcase the generative hallucinations we want to study.
  - Generative models (FLAN, Mistral, Llama) can follow context yet still invent details, which is the behaviour the cited papers target.

The code supports additional llama.cpp models when you set `MISTRAL_GGUF_PATH` and `LLAMA3_GGUF_PATH`, but the current exploration focuses on FLAN for simplicity.

## Running Experiments
1. Activate the virtual environment and install requirements:
   ```bash
   source venv/bin/activate
   python -m pip install -r requirements.txt
   ```
2. Execute the QA run:
   ```bash
   python main.py
   ```
   - Answers + detector labels print to stdout.
   - Results are appended to `runs/run-<timestamp>.jsonl`.
3. Generate visuals:
   ```bash
   python visualize_runs.py
   ```
   Outputs land in `runs/plots/`, including PNGs and a CSV mapping question indices to text.

## Detectors
Two detectors run on every result:

### Context Overlap (`detectors/context_overlap.py`)
- Tokenizes answer and context, counts overlapping words.
- Computes `support_ratio = overlap_tokens / answer_tokens`.
- Threshold currently `0.25`.
  - `ratio >= 0.25` → label **support**.
  - `ratio < 0.25` → label **flag**.
  - If either side lacks tokens → label **insufficient**.
- Interpretation: how much the answer reuses vocabulary from the evidence. High support suggests grounding; low support indicates the model introduced new terminology.

### Self Consistency (`detectors/self_consistency.py`)
- Calls `sample_answers()` on the model to draw **5** stochastic generations (temperature 0.6).
- Normalizes replies, counts unique variations, and divides by total samples to get a diversity score.
- Threshold currently `0.4`.
  - `diversity > 0.4` → label **flag** (model is unstable).
  - `diversity <= 0.4` → label **consistent**.
  - If the backend cannot sample → label **unsupported**.
- Interpretation: if multiple runs disagree, the answer is unreliable even when the first pass looks plausible.

Combining both detectors helps separate *ungrounded* answers from *unstable* ones.

## Flags vs. Support vs. Consistent
- **flag**: detector believes the answer is risky (either little lexical support or high disagreement).
- **support**: sufficient lexical overlap with the provided context.
- **consistent**: sampled answers largely agree with each other.
- **insufficient/unsupported**: detector could not evaluate (missing tokens or model lacks sampler).

## Visualizations
Generated plots (all under `runs/plots/`):

1. **`*_detector_counts.png`** – Bar chart showing how many times each detector emitted `support`, `flag`, or `consistent`. Quick sanity check for overall behaviour.
2. **`*_overlap_vs_consistency.png`** – Scatter with slight jitter. Red points are questions flagged by any detector. Labels `Q#` correspond to rows in `*_question_index.csv`. Use this to compare grounding (x-axis) and stability (y-axis).
3. **`*_flagged_questions.png`** – Horizontal bar plot listing question indices ordered by how many detectors flagged them (0, 1, or 2). Helps prioritise manual review.
4. **`*_flag_matrix.png`** – Heatmap of detectors (columns) vs. questions (rows). Cells show `1` when the detector flagged that question. Provides a compact overview across detectors.
5. **`*_question_index.csv`** – Lookup table mapping `qid` to the actual question text for referencing plotted labels.

## Extending the Pipeline
- Add or modify prompts in `main.py` to target new behaviours; JSONL and plots adjust automatically.
- Tune detector thresholds or sampling temperatures to experiment with sensitivity.
- When ready, set environment variables for Mistral or Llama-3 GGUF files to compare models.
- Incorporate external datasets (e.g., AggreFact) by feeding their questions and contexts into `QUESTIONS`.

## References
- Huang et al. (2025) — hallucination survey for taxonomy and challenges.
- Kryscinski et al. (2020) — factual consistency metrics inspire the overlap detector.
- Manakul et al. (2023) — SelfCheckGPT motivates the self-consistency sampling strategy.

With these components, teammates can reproduce runs, understand the detector outputs, and build on the current hallucination baseline. Remember to keep files under 200 lines and document any new detectors or plots for consistency.

