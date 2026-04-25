# 📊 Datasets Used for Main Project

This project evaluates hallucination detection across **multiple domains and hallucination types**, ensuring robustness and cross-domain generalisation.

---

## 🧠 1. HaluEval Benchmark

**Primary dataset for training and evaluation**

* 📌 Source: Li et al., *EMNLP 2023*
* 🔗 https://huggingface.co/datasets/pminervini/HaluEval

### 📂 Domains Used

| Domain                      | Description                          | Usage                  |
| --------------------------- | ------------------------------------ | ---------------------- |
| **QA (Question Answering)** | Knowledge-grounded factual responses | Training + Validation  |
| **Dialogue**                | Conversational responses             | Training + OOD Testing |
| **Summarisation**           | Abstractive summaries of documents   | OOD Testing            |

---

### 🧾 Data Format

Each example contains:

* **Context / Knowledge**
* **Input Prompt**
* **Truthful Response**
* **Hallucinated Response**

This allows creation of **counterfactual pairs**:

```id="1q7x2p"
(x_true, x_hallucinated)
```

---

### 📊 Dataset Statistics

| Split                | Size   |
| -------------------- | ------ |
| QA Train             | 14,000 |
| QA Validation        | 2,000  |
| QA Test              | 2,000  |
| Dialogue (Test)      | 2,000  |
| Summarisation (Test) | 2,000  |

---

## 🎯 2. TruthfulQA

**Benchmark for misconception-based hallucination**

* 📌 Source: Lin et al., *ACL 2022*
* 🔗 https://huggingface.co/datasets/truthfulqa/truthful_qa

---

### 📂 Description

TruthfulQA is designed to test whether models:

* mimic **human misconceptions**
* generate **confident but incorrect answers**

---

### 📊 Dataset Details

| Attribute        | Value                 |
| ---------------- | --------------------- |
| Total Questions  | 817                   |
| Evaluation Split | MC1 (Multiple Choice) |
| Final Samples    | 1,634 (balanced)      |

---

### ⚠️ Why It Matters

Unlike standard hallucination:

* ❌ Not random errors
* ✅ Systematic misconceptions

➡️ This creates **anti-correlation in residual signals**, which standard detectors fail on.

---

## 🔁 3. Counterfactual Dataset Construction

A key component of this project is **counterfactual analysis**.

For each dataset:

* Construct pairs:

```id="6uv4j2"
(x_true, x_false)
```

* Compute:

```id="g8k3s1"
Δ = features(x_true) - features(x_false)
```

This enables:

* ✔ causal effect estimation
* ✔ invariant feature selection (ICFS v2)

---

## 🌐 4. Domain Coverage

The dataset suite spans **four fundamentally different hallucination types**:

| Domain        | Type of Hallucination      |
| ------------- | -------------------------- |
| QA            | Factual incorrectness      |
| Dialogue      | Context inconsistency      |
| Summarisation | Lack of source grounding   |
| TruthfulQA    | Misconception-based errors |

---

## 🚀 Summary

This dataset combination enables:

* ✔ Cross-domain evaluation
* ✔ Causal feature discovery
* ✔ Robust generalisation testing
* ✔ Identification of failure modes in LLMs

---

## 📌 Citation

```bibtex id="k8x2dp"
@inproceedings{li2023halueval,
  title={HaluEval: A Large-Scale Hallucination Evaluation Benchmark},
  author={Li et al.},
  booktitle={EMNLP},
  year={2023}
}

@inproceedings{lin2022truthfulqa,
  title={TruthfulQA: Measuring How Models Mimic Human Falsehoods},
  author={Lin et al.},
  booktitle={ACL},
  year={2022}
}
```
