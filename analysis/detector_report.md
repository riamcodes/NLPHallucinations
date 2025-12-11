# Hallucination Detector Evaluation Report

**Model:** `google/flan-t5-base`  
**Run file:** `runs/run-20251211-061701.jsonl`  
**Gold labels:** `data/gold_labels.csv`  
---

## 1. Goals

This part of the project focuses on evaluating and extending hallucination detection for a generative QA model (FLAN-T5-Base). The main goals were:

- To measure how well different detectors can identify hallucinations using gold labels created by Person A.
- To understand where detectors succeed or fail across different question categories (simple factual, commonsense, negation adversarial, impossible/nonsense, obscure/long-tail factual, time-sensitive, and context-dependent).
- To design and test an ensemble that combines multiple detectors into a single hallucination score.
- To document limitations and propose concrete directions for improving hallucination detection in future work.

---

## 2. Detectors Implemented

We evaluated five detectors in total: three baselines and two new Person-B detectors.

### 2.1 Context Overlap (baseline)

- **Idea:** Compare the lexical overlap between the model’s answer and the provided context.
- **Intuition:** If the answer shares almost no tokens with the context, it is less likely to be grounded.
- **Labels:** `support`, `flag`, `insufficient`.
- **Observation:** In this run, most questions do not have rich context, so this detector almost never triggers a flag, which leads to high accuracy but zero recall.

### 2.2 Self Consistency (baseline)

- **Idea:** Sample multiple answers at a fixed temperature and compute how often the model disagrees with itself.
- **Intuition:** If the model gives very different answers to the same question, it is less trustworthy and more likely to hallucinate.
- **Labels:** `consistent`, `flag`.
- **Behavior:** This detector is very aggressive in this configuration: it flags almost all hallucinations, but also flags many correct answers.

### 2.3 Context Ablation (baseline)

- **Idea:** Remove the most supportive sentence from the context and regenerate the answer.
- **Intuition:** If the answer does not change when key evidence is removed, the model might be relying on parametric knowledge instead of the provided context.
- **Labels:** `context_sensitive`, `flag_context_ignored`, `insufficient`.
- **Observation:** Given that most questions in this dataset are not deeply context-heavy, this detector rarely produces `flag` and is effectively inactive on this run.

### 2.4 Temperature Sensitivity (new)

- **File:** `detectors/temperature_sensitivity.py`  
- **Idea:** Sample answers at multiple temperatures and measure how much they drift away from the baseline answer.
- **Implementation details:**
  - Temperatures: `[0.3, 0.7, 1.0]`
  - Samples per temperature: `2`
  - For each sample, compute a Jaccard-like word overlap with the baseline answer.
  - Define **drift = 1 – overlap**, and compute mean and max drift across all samples.
- **Decision rule:**  
  - Flag as hallucination when:
    - `mean_drift ≥ 0.35` **or**
    - `max_drift ≥ 0.60`.
- **Labels:** `temperature_stable`, `flag_temp_sensitive`, `insufficient`.
- **Intuition:** Large changes in the answer when we increase temperature indicate unstable, low-confidence behavior that correlates with hallucinations.

### 2.5 Semantic Overlap (new)

- **File:** `detectors/semantic_overlap.py`  
- **Idea:** Replace simple token overlap with an embedding-based similarity between the answer and the context.
- **Implementation details:**
  - Model: `sentence-transformers/all-MiniLM-L6-v2`
  - We encode `[context, answer]` and use cosine similarity between the two normalized embeddings.
  - If the similarity is below a threshold, we treat the answer as semantically unsupported.
- **Decision rule:**  
  - Flag as hallucination when similarity `< 0.60`.
- **Labels:** `support`, `flag`, `unsupported`, `insufficient`.
- **Intuition:** Even if some words overlap, the answer can still be semantically off. Embeddings capture paraphrases and looser lexical matches.

---

## 3. Overall Metrics

We treat any label that starts with `"flag"` (e.g., `flag`, `flag_temp_sensitive`, `flag_context_ignored`) as predicting hallucination, and anything else as predicting non-hallucination.

### 3.1 Overall Detector and Ensemble Performance

**N = 69 questions**

| name                         |  N |  acc  |  prec | recall |   F1  |
|------------------------------|---:|------:|------:|------:|------:|
| det_context_ablation         | 69 | 0.826 | 0.000 | 0.000 | 0.000 |
| det_context_overlap          | 69 | 0.826 | 0.000 | 0.000 | 0.000 |
| det_self_consistency         | 69 | 0.290 | 0.197 | 1.000 | 0.329 |
| det_semantic_overlap         | 69 | 0.754 | 0.000 | 0.000 | 0.000 |
| det_temperature_sensitivity  | 69 | 0.333 | 0.207 | 1.000 | 0.343 |
| ens_all                      | 69 | 0.826 | 0.000 | 0.000 | 0.000 |
| ens_any                      | 69 | 0.188 | 0.176 | 1.000 | 0.300 |
| ens_majority                 | 69 | 0.377 | 0.218 | 1.000 | 0.358 |
| **ens_weighted**             | 69 | 0.812 | 0.000 | 0.000 | 0.000 |

**Key takeaways:**

- **Self Consistency** and **Temperature Sensitivity** both achieve **perfect recall (1.0)** but have **low precision (~0.20)**. They catch virtually all hallucinations but over-flag many safe answers.
- **Context Overlap**, **Context Ablation**, and **Semantic Overlap** barely ever fire in this setting. They behave like “always predict non-hallucination” and therefore have:
  - High accuracy (≈0.75–0.83), matching the majority-class baseline.
  - Zero precision and recall on hallucinations.
- **Ensemble ANY** and **Ensemble Majority** inherit the high-recall / low-precision behavior from the aggressive detectors:
  - Both reach **recall = 1.0** but at the cost of very low accuracy (0.188 and 0.377 respectively).
- The **Weighted Ensemble** (`ens_weighted`) is currently **too conservative**:
  - With the current weights and threshold (0.5), it never reaches the threshold on any example, so it predicts “no hallucination” for all items.
  - This yields high accuracy (0.812) but **recall = 0.0**, and is effectively just a majority-class baseline.

---

## 4. Category-Level Behavior

We also break metrics down by the gold label categories.

### 4.1 Impossible / Nonsense Questions

These questions are deliberately unanswerable or conceptually incoherent, so the gold labels mark them as hallucinations.

| name                        |  N |  acc  |  prec | recall |   F1  |
|-----------------------------|---:|------:|------:|------:|------:|
| det_context_ablation        | 12 | 0.000 | 0.000 | 0.000 | 0.000 |
| det_context_overlap         | 12 | 0.000 | 0.000 | 0.000 | 0.000 |
| det_self_consistency        | 12 | 1.000 | 1.000 | 1.000 | 1.000 |
| det_semantic_overlap        | 12 | 0.000 | 0.000 | 0.000 | 0.000 |
| det_temperature_sensitivity | 12 | 1.000 | 1.000 | 1.000 | 1.000 |
| ens_all                     | 12 | 0.000 | 0.000 | 0.000 | 0.000 |
| ens_any                     | 12 | 1.000 | 1.000 | 1.000 | 1.000 |
| ens_majority                | 12 | 1.000 | 1.000 | 1.000 | 1.000 |
| ens_weighted                | 12 | 0.000 | 0.000 | 0.000 | 0.000 |

**Interpretation:**

- **Self Consistency** and **Temperature Sensitivity** are **perfect** on impossible/nonsense questions: they always flag them, and those flags always correspond to gold hallucinations.
- Simple lexical and semantic overlap with context are not useful here because these questions have **no meaningful context to compare against**, so they never flag.
- The **any** and **majority** ensembles perform **perfectly** on this category, since these questions are exactly the kind where instability + temperature drift show up strongly.
- The **weighted ensemble**, in contrast, never flags here, which is clearly undesirable for this category.

### 4.2 Simple Factual, Commonsense, Negation Adversarial, Obscure Long-Tail, Time-Sensitive

For most of the remaining categories, the detectors either:

- behave like majority-class baselines (never flag), or
- flag inconsistently, resulting in many false positives and zero true positives in that category.

Example patterns:

- In **simple_factual**, **commonsense**, **negation_adversarial**, **obscure_long_tail_factual**, and **time_sensitive**, we see:
  - Accuracy often **1.0** for context-based detectors and for `ens_weighted`, but **precision/recall = 0.0**, which means they are simply not detecting hallucinations in these settings.
  - The hallucinations in these categories tend to be **plausible-sounding but wrong**, so signal from self-consistency and temperature sensitivity is noisy and not cleanly aligned with gold labels.

### 4.3 Context-Dependent Questions

Context-dependent questions are the one place where semantic similarity should matter the most, because the answer must be supported by a provided paragraph.

| name                        |  N |  acc  |  prec | recall |   F1  |
|-----------------------------|---:|------:|------:|------:|------:|
| det_context_ablation        |  5 | 1.000 | 0.000 | 0.000 | 0.000 |
| det_context_overlap         |  5 | 1.000 | 0.000 | 0.000 | 0.000 |
| det_self_consistency        |  5 | 0.800 | 0.000 | 0.000 | 0.000 |
| det_semantic_overlap        |  5 | 0.000 | 0.000 | 0.000 | 0.000 |
| det_temperature_sensitivity |  5 | 0.800 | 0.000 | 0.000 | 0.000 |
| ens_all                     |  5 | 1.000 | 0.000 | 0.000 | 0.000 |
| ens_any                     |  5 | 0.000 | 0.000 | 0.000 | 0.000 |
| ens_majority                |  5 | 0.800 | 0.000 | 0.000 | 0.000 |
| ens_weighted                |  5 | 0.800 | 0.000 | 0.000 | 0.000 |

**Interpretation:**

- In this dataset, all context-dependent examples are **answered correctly by the model**, so the gold labels are non-hallucinations.
- However, `semantic_overlap` uses a fairly strict similarity threshold (0.60), and in these runs it **flags all of the correct context-dependent answers** as unsupported, leading to:
  - **Accuracy = 0.0** (every prediction is wrong),
  - no true positives (because there are no hallucinations in this subset),
  - and effectively showing that the current threshold is too harsh in this setting.
- Context overlap and context ablation both score 1.0 accuracy because they simply never fire `flag` for these examples.

---

## 5. Interpretation & Failure Modes

From these results, we can identify several key patterns:

### 5.1 “Always-Safe” Detectors

- **Context Overlap**, **Context Ablation**, and (with current thresholds) **Semantic Overlap** are effectively “always safe” in this run:
  - They almost never raise a `flag`.
  - Their accuracy is therefore close to the majority-class baseline (never predict hallucination).
- This makes them **high-precision but zero-recall** detectors. They are not useful on their own if we care about catching hallucinations.

### 5.2 High-Recall but Noisy Detectors

- **Self Consistency** and **Temperature Sensitivity** both achieve **recall = 1.0** overall.
- They are very good at **not missing hallucinations**, especially on **impossible/nonsense** questions.
- However, they tend to **over-flag**:
  - Many correct answers also look unstable or temperature-sensitive.
  - This leads to precision around 0.20 and low overall accuracy (0.29 and 0.33).

### 5.3 Stable Hallucinations

- Some hallucinations remain **very stable** across samples and temperatures.
- In those cases, even the high-recall detectors can be misleading:
  - Self-consistency may still flag them if minor wording differences show up, but the underlying *fact* stays the same and wrong.
  - If we tuned them to be less sensitive, these stable hallucinations would be missed entirely.

### 5.4 Semantic Thresholding Issues

- The current semantic similarity threshold (0.60) is too aggressive for the context-dependent examples:
  - Even clearly supported answers (e.g., “Canada, Japan, and Brazil” given a paragraph explicitly naming those countries) can fall below this threshold.
- This highlights how **absolute embedding similarity thresholds** are fragile:
  - They depend heavily on the model, the domain, and the length of the context.

---

## 6. Ensemble Design & Behavior

We evaluated four ensembles:

1. **ens_any** – predict hallucination if *any* detector flags.
2. **ens_majority** – predict hallucination if ≥ half of detectors flag.
3. **ens_all** – predict hallucination only if *all* detectors flag.
4. **ens_weighted** – new weighted ensemble combining detector outputs.

### 6.1 Logical Ensembles (any / majority / all)

- **ens_any** and **ens_majority** inherit the behaviour of the high-recall detectors:
  - Both achieve **recall = 1.0**, but with very poor accuracy (0.188 and 0.377).
  - These are appropriate only if we care **almost exclusively** about recall and are willing to tolerate many false positives (e.g., safety-critical settings where a human reviews all flags).
- **ens_all** behaves like the conservative detectors:
  - With the current thresholds and detector behaviour, it almost never predicts hallucination, so it degenerates into the majority baseline (0.826 accuracy, zero recall).

### 6.2 Weighted Ensemble

We define a weighted ensemble score:

\[
\text{score} = 0.25 \cdot \text{context\_overlap} +
               0.25 \cdot \text{self\_consistency} +
               0.20 \cdot \text{context\_ablation} +
               0.15 \cdot \text{temperature\_sensitivity} +
               0.15 \cdot \text{semantic\_overlap}
\]

where each detector contributes its weight only when it predicts a `flag`. We then label hallucination when:

\[
\text{score} \ge 0.5.
\]

**Observed behavior:**

- In this run, **no example ever reaches the threshold of 0.5**, so `ens_weighted` always predicts “no hallucination.”
- This yields **accuracy ≈ 0.81**, which matches the proportion of non-hallucinations in the dataset, but **recall = 0**, so it is **not a useful detector** in its current form.
- This is a clear sign that the combination of:
  - conservative thresholds in some detectors, and
  - a relatively high ensemble threshold
  makes the ensemble overly cautious.

---

## 7. Limitations

Several limitations of the current detector setup emerged from the experiments:

1. **Small dataset size**  
   - With only 69 questions, per-category metrics are sensitive to a few examples.
   - A single miscalibrated threshold can cause entire categories (e.g., context-dependent) to be misclassified.

2. **Lack of rich context for most questions**  
   - Many questions (simple factual, commonsense, time-sensitive) have no real context passage.
   - This makes context-based detectors (overlap, ablation, semantic overlap) effectively inactive.

3. **Hand-tuned thresholds**  
   - Thresholds for semantic similarity and temperature drift were chosen heuristically, not learned.
   - As a result, some detectors are either too aggressive (self-consistency, temperature) or too conservative (semantic overlap, context overlap).

4. **Single model architecture**  
   - All results are for `google/flan-t5-base` only.
   - We do not yet know how these detectors behave on other models (e.g., Llama, Mistral), or whether the same thresholds would transfer.

---

## 8. Proposed Improvements & Future Work

Based on these findings, we propose several directions for improving hallucination detection in this project:

### 8.1 Threshold Tuning and Calibration

- **Semantic Overlap**
  - Lower the similarity threshold (e.g., from 0.60 to something like 0.40–0.45) for context-dependent questions, or make the threshold adaptive based on context length.
- **Temperature Sensitivity & Self Consistency**
  - Slightly relax the conditions for flagging (e.g., require both high mean drift and high max drift) to reduce false positives on obviously correct factual answers.

### 8.2 Smarter Ensembles

- **Dynamic weighting**
  - Instead of fixed weights, learn ensemble weights on a small validation set using logistic regression or another simple classifier.
- **Category-aware ensembles**
  - Use different detector subsets per category:
    - For **impossible/nonsense**, rely heavily on self-consistency and temperature sensitivity.
    - For **context-dependent** questions, rely more on semantic overlap and context ablation.
- **Re-tuned weighted ensemble**
  - Lower the global threshold (e.g., from 0.50 to 0.30) so that the ensemble actually produces some positive predictions.

### 8.3 Additional Detectors

- A **retrieval-based fact checker** that compares model answers against a trusted knowledge source (e.g., Wikipedia).
- A **contradiction detector** that checks whether the answer contradicts the context or the question wording (useful for negation adversarial questions).

### 8.4 Model Comparisons (Future Extra Credit)

- Run the full pipeline on another model (e.g., a Llama-based QA system) and compare:
  - overall hallucination rate,
  - which detectors work better for each model,
  - whether the same ensemble configuration transfers or needs to be re-tuned.

---

## 9. Summary

From Person B’s experiments, we learn that:

- **Instability-based detectors** (self-consistency and temperature sensitivity) are extremely effective on impossible/nonsense questions and achieve perfect recall overall, but they are noisy and over-flag many correct answers.
- **Context-based detectors** (overlap, ablation, semantic overlap) are severely underutilized on this dataset due to limited context and conservative thresholds.
- Simple OR/majority ensembles are good “safety nets” when recall is the priority, while the current weighted ensemble is too conservative and behaves like a majority-class baseline.
- Good hallucination detection requires **careful threshold tuning, category-aware design, and potentially learned ensembles**, not just stacking detectors together.

This report and the associated scripts (`detectors/temperature_sensitivity.py`, `detectors/semantic_overlap.py`, and `analysis/evaluate_detectors.py`) document the current behaviour and establish a baseline for future improvements.
