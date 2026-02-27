#!/usr/bin/env python3
"""
Extract SAE features from Gemma-2-2B for all dataset splits.

Standalone script for feature extraction. Useful when running on
GPU instances with time limits (e.g., Kaggle 9-hour sessions).

Usage:
    python scripts/extract_features.py --config configs/default.yaml
    python scripts/extract_features.py --splits qa_train,qa_test,dial_test
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from musae_inv.config import Config
from musae_inv.utils.seed import set_seed
from musae_inv.utils.io import save_pickle, load_pickle
from musae_inv.data.datasets import load_all_datasets
from musae_inv.features.extraction import extract_features
from musae_inv.features.counterfactual import extract_counterfactual_pairs
from musae_inv.models.model_loader import load_gemma_model, load_gemma_scope_saes


def parse_args():
    parser = argparse.ArgumentParser(description="Feature Extraction")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--splits", type=str, default="all",
                        help="Comma-separated split names or 'all'")
    parser.add_argument("--counterfactual", action="store_true",
                        help="Also extract counterfactual deltas")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.config:
        cfg = Config.load(args.config)
    else:
        cfg = Config(output_dir=args.output_dir)
    cfg.make_dirs()
    set_seed(cfg.seed)

    print("=" * 70)
    print("  MuSAE-Inv Feature Extraction")
    print("=" * 70)

    t_start = time.time()

    # Load model and SAEs
    model, tokenizer = load_gemma_model(cfg)
    saes = load_gemma_scope_saes(cfg)

    # Load datasets
    datasets = load_all_datasets(cfg)

    # Define splits
    all_splits = [
        ("halueval_qa_train", datasets["halueval_qa_train"]),
        ("halueval_qa_val", datasets["halueval_qa_val"]),
        ("halueval_qa_test", datasets["halueval_qa_test"]),
        ("halueval_dial_test", datasets["halueval_dial_test"]),
        ("halueval_summ_test", datasets["halueval_summ_test"]),
        ("truthfulqa", datasets["truthfulqa"]),
    ]

    if args.splits != "all":
        requested = set(args.splits.split(","))
        all_splits = [(n, d) for n, d in all_splits if n in requested]

    # Extract
    feat_cache_path = cfg.features_dir / "feature_cache.pkl"
    if feat_cache_path.exists():
        feature_cache = load_pickle(feat_cache_path)
        print(f"Loaded existing cache ({len(feature_cache)} splits)")
    else:
        feature_cache = {}

    for name, df in all_splits:
        if name in feature_cache:
            print(f"  {name}: already cached, skipping")
            continue
        print(f"\n  Extracting: {name} ({len(df)} examples)")
        feature_cache[name] = extract_features(
            df, model, tokenizer, saes, cfg.target_layers, cfg=cfg, desc=name,
        )
        save_pickle(feature_cache, feat_cache_path)
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n✅ Feature extraction complete ({time.time() - t_start:.0f}s)")

    # Counterfactual extraction
    if args.counterfactual:
        print("\n  Extracting counterfactual deltas...")
        cf_cache_path = cfg.features_dir / "counterfactual_cache.pkl"
        cf_cache = load_pickle(cf_cache_path) if cf_cache_path.exists() else {}

        for domain, key, n_pairs in [
            ("qa", "halueval_qa_pairs", cfg.n_cf_qa),
            ("dialogue", "halueval_dial_pairs", cfg.n_cf_dial),
            ("summarisation", "halueval_summ_pairs", cfg.n_cf_summ),
        ]:
            if domain in cf_cache:
                print(f"    {domain}: cached, skipping")
                continue
            pairs = datasets[key].iloc[:n_pairs]
            print(f"    {domain}: {len(pairs)} pairs")
            ft, ff, delta = extract_counterfactual_pairs(
                pairs, model, tokenizer, saes, cfg.target_layers, cfg=cfg, desc=domain,
            )
            cf_cache[domain] = {"feat_true": ft, "feat_false": ff, "delta": delta}
            del ft, ff
            gc.collect()
            torch.cuda.empty_cache()
            save_pickle(cf_cache, cf_cache_path)

        print("✅ Counterfactual extraction complete.")


if __name__ == "__main__":
    main()
