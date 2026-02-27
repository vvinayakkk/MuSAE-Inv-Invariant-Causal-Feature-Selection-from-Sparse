# Datasets

MuSAE-Inv evaluates cross-domain hallucination detection using four benchmarks spanning three domains.

## HaluEval (Li et al., 2023)

**Paper**: [HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models](https://aclanthology.org/2023.emnlp-main.397/)  
**HuggingFace**: [pminervini/HaluEval](https://huggingface.co/datasets/pminervini/HaluEval)  
**License**: MIT  
**GitHub**: [RUCAIBox/HaluEval](https://github.com/RUCAIBox/HaluEval)

HaluEval is a large-scale benchmark for evaluating hallucination detection capabilities. It contains ChatGPT-generated hallucinated outputs paired with correct responses, enabling controlled evaluation of hallucination detectors.

### HaluEval QA (Question Answering)

| Field | Details |
|:---|:---|
| **Split name** | `qa_samples` |
| **Size** | 10,000 examples |
| **Usage** | In-distribution training + test |
| **Task** | Knowledge-grounded QA |
| **Columns** | `knowledge`, `question`, `right_answer`, `hallucinated_answer` |

In our pipeline, we use 70/15/15 train/val/test splits (7000/1500/1500).

### HaluEval Dialogue

| Field | Details |
|:---|:---|
| **Split name** | `dialogue_samples` |
| **Size** | 1,000 examples |
| **Usage** | Out-of-distribution test |
| **Task** | Knowledge-grounded dialogue |
| **Columns** | `knowledge`, `dialogue_history`, `right_response`, `hallucinated_response` |

### HaluEval Summarisation

| Field | Details |
|:---|:---|
| **Split name** | `summarization_samples` |
| **Size** | 1,000 examples |
| **Usage** | Out-of-distribution test |
| **Task** | Document summarisation |
| **Columns** | `document`, `right_summary`, `hallucinated_summary` |

## TruthfulQA (Lin et al., 2022)

**Paper**: [TruthfulQA: Measuring How Models Mimic Human Falsehoods](https://aclanthology.org/2022.acl-long.229/)  
**HuggingFace**: [truthfulqa/truthful_qa](https://huggingface.co/datasets/truthfulqa/truthful_qa)  
**License**: Apache-2.0  
**GitHub**: [sylinrl/TruthfulQA](https://github.com/sylinrl/TruthfulQA)

| Field | Details |
|:---|:---|
| **Configuration** | `multiple_choice` |
| **Split** | `validation` |
| **Size** | 817 questions |
| **Usage** | Out-of-distribution test (zero-shot) |
| **Task** | General knowledge truthfulness |
| **Categories** | 38 categories including health, law, finance, politics |

TruthfulQA tests whether models generate truthful answers to questions designed to elicit common human misconceptions. We use the `mc1_targets` (single-true) format.

## Data Preparation

### Downloading

```bash
# Automatic download via HuggingFace datasets
python scripts/download_data.py

# Downloads are cached in ~/.cache/huggingface/
```

### Text Construction

For each dataset, we construct a unified text representation:

```python
# QA
text = f"Question: {row['question']}\nAnswer: {row['answer']}"

# Dialogue  
text = f"Dialogue: {row['dialogue_history']}\nResponse: {row['response']}"

# Summarisation
text = f"Document: {row['document'][:500]}\nSummary: {row['summary']}"

# TruthfulQA
text = f"Question: {row['question']}\nAnswer: {row['best_answer']}"
```

### Split Strategy

```
HaluEval QA:
  Training:   7,000 (70%) — used for probe training
  Validation: 1,500 (15%) — used for early stopping
  Test:       1,500 (15%) — in-distribution evaluation

HaluEval Dialogue: 1,000 — OOD test only
HaluEval Summarisation: 1,000 — OOD test only
TruthfulQA: 817 — OOD test only (zero-shot transfer)
```

### Counterfactual Pairs

For ICFS scoring, we construct (true, hallucinated) pairs:

```
QA pairs:         3,000 (from training set)
Dialogue pairs:   1,500 
Summarisation:    1,500
```

## Schema Detection

The codebase includes robust schema detection (`_find_col()`) that handles multiple column naming conventions across HaluEval versions:

```python
# Handles: "right_answer", "correct_answer", "answer", "response", etc.
# Automatically detects hallucinated columns regardless of naming convention
```
