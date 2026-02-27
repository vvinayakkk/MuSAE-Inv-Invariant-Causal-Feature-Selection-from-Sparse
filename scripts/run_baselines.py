#!/usr/bin/env python3
"""
Run all baseline methods for comparison.

Separates baseline evaluation from the main training pipeline
for modularity and faster iteration.

Usage:
    python scripts/run_baselines.py --output-dir ./outputs
"""

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from musae_inv.config import Config
from musae_inv.utils.seed import set_seed
from musae_inv.utils.io import load_pickle
from musae_inv.data.preprocessing import build_icfs_features
from musae_inv.evaluation.metrics import ResultsTracker
from musae_inv.evaluation.baselines import sae_entropy_score, tdv_direction_vector, tdv_score
from musae_inv.models.probes import SingleLayerProbe, ConcatPCAProbe
from musae_inv.models.saplma import train_saplma_probe
from musae_inv.visualization.tables import generate_result_tables


def parse_args():
    parser = argparse.ArgumentParser(description="Run All Baselines")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.load(args.config) if args.config else Config(output_dir=args.output_dir)
    set_seed(cfg.seed)

    print("Loading caches...")
    feature_cache = load_pickle(cfg.features_dir / "feature_cache.pkl")

    y_train = feature_cache["halueval_qa_train"]["labels"]
    y_val = feature_cache["halueval_qa_val"]["labels"]
    y_test = feature_cache["halueval_qa_test"]["labels"]
    y_dial = feature_cache["halueval_dial_test"]["labels"]
    y_summ = feature_cache["halueval_summ_test"]["labels"]
    y_tqa = feature_cache["truthfulqa"]["labels"]

    tracker = ResultsTracker()
    rng = np.random.RandomState(cfg.seed)

    # Random
    print("\n── Random ──")
    for dom, y in [("QA", y_test), ("Dialogue", y_dial), ("Summ", y_summ), ("TruthfulQA", y_tqa)]:
        tracker.register("Random", dom, y, rng.uniform(0, 1, len(y)))

    # SelfCheckGPT-NLI proxy
    print("\n── SelfCheckGPT-NLI proxy ──")
    fc_map = {"QA": "halueval_qa_test", "Dialogue": "halueval_dial_test",
              "Summ": "halueval_summ_test", "TruthfulQA": "truthfulqa"}
    for dom, y in [("QA", y_test), ("Dialogue", y_dial), ("Summ", y_summ), ("TruthfulQA", y_tqa)]:
        scores = sae_entropy_score(feature_cache[fc_map[dom]], cfg.target_layers)
        tracker.register("SelfCheckGPT-NLI proxy", dom, y, scores)

    # SAPLMA
    print("\n── SAPLMA ──")
    saplma_probe, saplma_scaler = train_saplma_probe(
        feature_cache["halueval_qa_train"]["sae_features"][18],
        y_train,
        feature_cache["halueval_qa_val"]["sae_features"][18],
        y_val,
        cfg,
    )
    for dom, fc_key, y in [("QA", "halueval_qa_test", y_test),
                            ("Dialogue", "halueval_dial_test", y_dial),
                            ("Summ", "halueval_summ_test", y_summ),
                            ("TruthfulQA", "truthfulqa", y_tqa)]:
        X_sc = saplma_scaler.transform(feature_cache[fc_key]["sae_features"][18])
        with torch.no_grad():
            p = saplma_probe(torch.FloatTensor(X_sc).to(cfg.device)).cpu().numpy()
        tracker.register("SAPLMA (SAE-L18 MLP)", dom, y, p)
    del saplma_probe
    gc.collect()

    # Concat-4L + PCA + LR
    print("\n── Concat-4L + PCA + LR ──")
    X_cat_tr = np.concatenate([feature_cache["halueval_qa_train"]["sae_features"][l]
                                for l in cfg.target_layers], axis=1)
    concat_probe = ConcatPCAProbe(cfg=cfg)
    concat_probe.fit(X_cat_tr, y_train)
    for dom, fc_key, y in [("QA", "halueval_qa_test", y_test),
                            ("Dialogue", "halueval_dial_test", y_dial),
                            ("Summ", "halueval_summ_test", y_summ),
                            ("TruthfulQA", "truthfulqa", y_tqa)]:
        X_cat = np.concatenate([feature_cache[fc_key]["sae_features"][l]
                                for l in cfg.target_layers], axis=1)
        tracker.register("Concat-4L + PCA + LR", dom, y, concat_probe.predict_proba(X_cat))

    # Single-layer probes
    print("\n── Single-Layer Probes ──")
    for l in cfg.target_layers:
        sl = SingleLayerProbe(l, cfg)
        sl.fit(feature_cache["halueval_qa_train"]["sae_features"][l], y_train)
        for dom, fc_key, y in [("QA", "halueval_qa_test", y_test),
                                ("Dialogue", "halueval_dial_test", y_dial),
                                ("Summ", "halueval_summ_test", y_summ),
                                ("TruthfulQA", "truthfulqa", y_tqa)]:
            p = sl.predict_proba(feature_cache[fc_key]["sae_features"][l])
            tracker.register(f"SAE-L{l} LR", dom, y, p)

    # Save results
    generate_result_tables(tracker.results, cfg.results_dir)
    print("\n✅ All baselines complete.")


if __name__ == "__main__":
    main()
