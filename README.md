# NLP5 Hallucination Detection Baseline

This project explores hallucination detection for generative QA models. It currently focuses on the `google/flan-t5-base` model with optional support for llama.cpp backends (Mistral, Llama-3).

## Quick Start

```bash
source venv/bin/activate
python -m pip install -r requirements.txt
python main.py          # run QA + detectors
python visualize_runs.py  # generate plots for the latest run
```

Outputs are appended to `runs/run-<timestamp>.jsonl` and visualizations are saved in `runs/plots/`.

## Running on a fresh clone
1. Clone the repo and create/activate a virtual environment.
   ```bash
   git clone <repo-url>
   cd NLP5
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies.
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
3. Ensure you have a CUDA/MPS-capable device if desired; FLAN runs on CPU but is faster with GPU/MPS.
4. Optional: set environment variables if you have llama.cpp GGUF weights.
   ```bash
   export MISTRAL_GGUF_PATH=/path/to/mistral.gguf
   export LLAMA3_GGUF_PATH=/path/to/llama3.gguf
   ```
5. Run `python main.py` to produce answers and detector scores.
6. Run `python visualize_runs.py` to generate plots summarizing the latest run.

## Updating questions/contexts
- Edit the `QUESTIONS` list near the top of `main.py`.
- Keep contexts including the ground-truth facts so detectors can judge grounding.
- Re-run `python main.py` and `python visualize_runs.py` after making changes.

---

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
Three detectors currently run on every result. Each one targets a different hallucination signal: grounding, stability, and context dependence.

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

### Context Ablation (`detectors/context_ablation.py`)
- Splits the context into sentences.
- Identifies the most supporting sentence for the baseline answer.
- Removes that sentence to create an ablated context.
- Regenerates the answer using the ablated context.
- Computes answer similarity using token overlap:
   - sensitivity=1−overlap_ratio(baseline,ablated)
- Uses two thresholds:
   - support_threshold = 0.2 → was the removed sentence meaningful?
   - sensitivity_threshold = 0.3 → did the answer change?
**Labels:**
   - flag_context_ignored → The model gives the same answer even after removing the supporting sentence
   → suggests reliance on parametric knowledge, not context
   - context_sensitive → The answer changes when evidence is removed
   - insufficient → Context too short or no meaningful support detected
- Interpretation: this detector reveals whether the model actually uses the provided context. Answers that survive ablation unchanged often indicate hallucinated confidence.


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
5. **`*_flagged_samples.png` / `*_flagged_samples.csv`** – For every self-consistency flag, display the baseline answer alongside the unique stochastic generations that caused the disagreement.
6. **`*_question_index.csv`** – Lookup table mapping `qid` to the actual question text for referencing plotted labels.

## Extending the Pipeline
- Add or modify prompts in `main.py` to target new behaviours; JSONL and plots adjust automatically.
- Tune detector thresholds or sampling temperatures to experiment with sensitivity.
- When ready, set environment variables for Mistral or Llama-3 GGUF files to compare models.
- Incorporate external datasets (e.g., AggreFact) by feeding their questions and contexts into `QUESTIONS`.
- Explore additional detectors (entailment models, retrieval consistency, reference metrics, etc.) and append their scores to the JSONL/logging pipeline.

## References
- Huang et al. (2025) — hallucination survey for taxonomy and challenges.
- Kryscinski et al. (2020) — factual consistency metrics inspire the overlap detector.
- Manakul et al. (2023) — SelfCheckGPT motivates the self-consistency sampling strategy.

With these components, teammates can reproduce runs, understand detector outputs, and build on the current hallucination baseline. Remember to keep files under 200 lines and document any new detectors or plots for consistency.
# Simple Question Answering with DistilBERT and GPT-2



## What You Need

- Python 3.8+
- 2 packages: `torch` and `transformers`

## Quick Setup

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install torch transformers

# Run the system
python main.py
```

## Every Time You Want to Run It

```bash
# Navigate to project directory
cd /path/to/NLP5

# Activate virtual environment
source venv/bin/activate

# Run the system
python main.py
```

## Yiippie yippie yippie hopeefully it worked

The system will:
1. Load DistilBERT and GPT-2 models
2. Answer 2 sample questions
3. Generate additional text with GPT-2
4. Show you the results

## How It Works

- **1 file**: `main.py` (163 lines)
- **2 models**: DistilBERT for Q&A + GPT-2 for text generation
- **Simple**: Just load and ask questions

## Example Output

```
Question 1: What is the capital of France?
Answer: Paris
Confidence: 0.987

Generated Text:
Question: What is the capital of France?
Context: France is a country in Western Europe...
Answer: Paris
Additional information: France is known for its rich history...
Generated Length: 25 words

Question 2: Who wrote Romeo and Juliet?
Answer: William Shakespeare
Confidence: 0.923

Generated Text:
Question: Who wrote Romeo and Juliet?
Context: Romeo and Juliet is a tragedy...
Answer: William Shakespeare
Additional information: Shakespeare was an English playwright...
Generated Length: 22 words
```

## For Your Own Questions

Edit the `questions` list in `main.py`:

```python
questions = [
    {
        "question": "Your question here?",
        "context": "Your context here."
    }
]
```

## Troubleshooting

### "externally-managed-environment" Error
If you get this error when running `pip install`:
```bash
error: externally-managed-environment
```

**Solution:** Use the virtual environment setup above. This error happens when trying to install packages globally on macOS.

### Virtual Environment Not Working
If `source venv/bin/activate` doesn't work:
```bash
# Make sure you're in the right directory
cd /path/to/NLP5

# Check if venv folder exists
ls -la

# If venv doesn't exist, create it
python3 -m venv venv
source venv/bin/activate
```

### Models Not Downloading
If models fail to download:
- Check your internet connection
- Ensure you have enough disk space (1GB+)
- Try running the command again

**That's it! No complex setup, no multiple files, just simple Q&A with text generation.**
DistilBERT and vanilla BERT are still useful, but they’re limited to extraction: they pick spans from the context rather than generating new text. That has two implications for this project:
They don’t really “hallucinate” in the generative sense, because they’re constrained to reuse the source wording. They can mis-pick spans, but they can’t fabricate new claims, so they’re mainly good as a control baseline.
We can repurpose them as factuality checkers—e.g., to retrieve supporting evidence or to judge whether a claim is entailed by the source—but on their own they won’t give us the free-form answers we need to stress-test hallucination detection.
So keep them as lightweight extractive models if you want a contrast, but for the core experiments you’ll want at least one generative, instruction-tuned decoder (FLAN, Mistral, Llama-3, etc.) that can produce unconstrained answers where hallucinations actually appear.