"""
Dataset loading for HaluEval (QA, Dialogue, Summarisation) and TruthfulQA.

Handles schema detection across HuggingFace dataset versions and returns
standardised DataFrames with 'text' and 'label' columns.

Datasets
--------
- HaluEval QA: 10,000 question-answer pairs (knowledge + question + answer)
- HaluEval Dialogue: 10,000 dialogue-response pairs
- HaluEval Summarisation: 10,000 document-summary pairs
- TruthfulQA: ~817 questions × multiple choices (balanced binary)
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from datasets import load_dataset

from musae_inv.config import Config


def _find_col(df: pd.DataFrame, candidates: list[str], default=None) -> str | None:
    """Find the first matching column name from a list of candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return default


def load_halueval_qa(cfg: Config) -> Dict[str, pd.DataFrame]:
    """Load HaluEval QA dataset and split into train/val/test.

    Returns
    -------
    dict
        Keys: 'pairs', 'train', 'val', 'test' DataFrames.
    """
    print("Loading HaluEval QA...")
    hq = load_dataset("pminervini/HaluEval", "qa_samples", split="data")
    hq_df = pd.DataFrame(hq)

    q_col = _find_col(hq_df, ["question"])
    true_col = _find_col(hq_df, ["answer", "correct_answer", "right_answer"])
    false_col = _find_col(hq_df, ["hallucination", "hallucinated_answer", "incorrect_answer"])
    know_col = _find_col(hq_df, ["knowledge"], default=None)

    context = (hq_df[know_col].astype(str) + " ") if know_col else ""

    hq_df["text_true"] = context + hq_df[q_col].astype(str) + " " + hq_df[true_col].astype(str)
    hq_df["text_false"] = context + hq_df[q_col].astype(str) + " " + hq_df[false_col].astype(str)
    hq_df["pair_id"] = np.arange(len(hq_df))

    # Build flat labelled dataset
    true_rows = hq_df[[q_col, true_col, "text_true", "pair_id"]].copy()
    true_rows = true_rows.rename(columns={"text_true": "text"})
    true_rows["label"] = 0

    false_rows = hq_df[[q_col, false_col, "text_false", "pair_id"]].copy()
    false_rows = false_rows.rename(columns={"text_false": "text"})
    false_rows["label"] = 1

    full_qa = pd.concat([true_rows, false_rows]).sample(
        frac=1, random_state=cfg.seed
    ).reset_index(drop=True)

    n_total = len(full_qa)
    n_train = min(cfg.n_train_qa * 2, int(0.7 * n_total))
    n_val = min(cfg.n_val_qa * 2, int(0.15 * n_total))

    train = full_qa.iloc[:n_train]
    val = full_qa.iloc[n_train : n_train + n_val]
    test = full_qa.iloc[n_train + n_val : n_train + n_val + cfg.n_test_qa * 2]

    print(f"  QA: train={len(train)} val={len(val)} test={len(test)} pairs={len(hq_df)}")
    return {"pairs": hq_df, "train": train, "val": val, "test": test}


def load_halueval_dialogue(cfg: Config) -> Dict[str, pd.DataFrame]:
    """Load HaluEval Dialogue dataset.

    Returns
    -------
    dict
        Keys: 'pairs', 'train', 'test' DataFrames.
    """
    print("Loading HaluEval Dialogue...")
    hd = load_dataset("pminervini/HaluEval", "dialogue_samples", split="data")
    hd_df = pd.DataFrame(hd)

    ctx_col = _find_col(hd_df, ["dialogue_history"])
    true_col = _find_col(hd_df, ["response", "right_response"])
    false_col = _find_col(hd_df, ["hallucination", "hallucinated_response"])

    context = (hd_df[ctx_col].astype(str) + " ") if ctx_col else ""
    hd_df["text_true"] = context + hd_df[true_col].astype(str)
    hd_df["text_false"] = context + hd_df[false_col].astype(str)
    hd_df["pair_id"] = np.arange(len(hd_df))

    dial_true = hd_df.copy()
    dial_true["text"] = dial_true["text_true"]
    dial_true["label"] = 0

    dial_false = hd_df.copy()
    dial_false["text"] = dial_false["text_false"]
    dial_false["label"] = 1

    dial_all = pd.concat([dial_true, dial_false]).sample(
        frac=1, random_state=cfg.seed
    ).reset_index(drop=True)

    train = dial_all.iloc[: cfg.n_dial_train * 2]
    test = dial_all.iloc[cfg.n_dial_train * 2 : cfg.n_dial_train * 2 + cfg.n_dial_test * 2]

    print(f"  Dialogue: train={len(train)} test={len(test)} pairs={len(hd_df)}")
    return {"pairs": hd_df, "train": train, "test": test}


def load_halueval_summarisation(cfg: Config) -> Dict[str, pd.DataFrame]:
    """Load HaluEval Summarisation dataset.

    Returns
    -------
    dict
        Keys: 'pairs', 'train', 'test' DataFrames.
    """
    print("Loading HaluEval Summarisation...")
    hs = load_dataset("pminervini/HaluEval", "summarization_samples", split="data")
    hs_df = pd.DataFrame(hs)

    doc_col = _find_col(hs_df, ["document", "article"], default=None)
    true_col = _find_col(hs_df, ["summary", "right_summary"])
    false_col = _find_col(hs_df, ["hallucination", "hallucinated_summary"])

    context = (hs_df[doc_col].astype(str) + " ") if doc_col else ""
    hs_df["text_true"] = context + hs_df[true_col].astype(str)
    hs_df["text_false"] = context + hs_df[false_col].astype(str)
    hs_df["pair_id"] = np.arange(len(hs_df))

    summ_true = hs_df.copy()
    summ_true["text"] = summ_true["text_true"]
    summ_true["label"] = 0

    summ_false = hs_df.copy()
    summ_false["text"] = summ_false["text_false"]
    summ_false["label"] = 1

    summ_all = pd.concat([summ_true, summ_false]).sample(
        frac=1, random_state=cfg.seed
    ).reset_index(drop=True)

    train = summ_all.iloc[: cfg.n_summ_train * 2]
    test = summ_all.iloc[cfg.n_summ_train * 2 : cfg.n_summ_train * 2 + cfg.n_summ_test * 2]

    print(f"  Summarisation: train={len(train)} test={len(test)} pairs={len(hs_df)}")
    return {"pairs": hs_df, "train": train, "test": test}


def load_truthfulqa(cfg: Config) -> pd.DataFrame:
    """Load TruthfulQA (multiple choice) as balanced binary classification.

    Returns
    -------
    pd.DataFrame
        Balanced binary DataFrame with 'text' and 'label' columns.
    """
    print("Loading TruthfulQA (multiple_choice)...")
    tqa = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")

    rows = []
    for ex in tqa:
        q = ex["question"]
        choices = ex["mc1_targets"]["choices"]
        labels = ex["mc1_targets"]["labels"]
        for ch, lab in zip(choices, labels):
            rows.append({"text": q + " " + ch, "label": 1 - lab})

    tqa_df = pd.DataFrame(rows).sample(frac=1, random_state=cfg.seed).reset_index(drop=True)

    # Balance classes
    pos = tqa_df[tqa_df.label == 1]
    neg = tqa_df[tqa_df.label == 0]
    n_min = min(len(pos), len(neg))
    tqa_df = (
        pd.concat([pos.iloc[:n_min], neg.iloc[:n_min]])
        .sample(frac=1, random_state=cfg.seed)
        .reset_index(drop=True)
    )

    print(f"  TruthfulQA binary: {len(tqa_df)} examples (balanced)")
    return tqa_df


def load_all_datasets(cfg: Config) -> Dict[str, pd.DataFrame]:
    """Load all four datasets and return a unified dictionary.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys include: halueval_qa_pairs, halueval_qa_train, halueval_qa_val,
        halueval_qa_test, halueval_dial_pairs, halueval_dial_train,
        halueval_dial_test, halueval_summ_pairs, halueval_summ_train,
        halueval_summ_test, truthfulqa.
    """
    datasets = {}

    qa = load_halueval_qa(cfg)
    datasets["halueval_qa_pairs"] = qa["pairs"]
    datasets["halueval_qa_train"] = qa["train"]
    datasets["halueval_qa_val"] = qa["val"]
    datasets["halueval_qa_test"] = qa["test"]

    dial = load_halueval_dialogue(cfg)
    datasets["halueval_dial_pairs"] = dial["pairs"]
    datasets["halueval_dial_train"] = dial["train"]
    datasets["halueval_dial_test"] = dial["test"]

    summ = load_halueval_summarisation(cfg)
    datasets["halueval_summ_pairs"] = summ["pairs"]
    datasets["halueval_summ_train"] = summ["train"]
    datasets["halueval_summ_test"] = summ["test"]

    datasets["truthfulqa"] = load_truthfulqa(cfg)

    print("\n✅ All datasets loaded.")
    return datasets
