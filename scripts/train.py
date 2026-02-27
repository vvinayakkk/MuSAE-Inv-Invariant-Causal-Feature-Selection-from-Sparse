#!/usr/bin/env python3
"""
MuSAE-Inv Full Training Pipeline.

End-to-end script that runs the complete MuSAE-Inv experiment:
    1. Load Gemma-2-2B and Gemma Scope SAEs
    2. Load HaluEval + TruthfulQA datasets
    3. Extract SAE features for all splits
    4. Extract counterfactual deltas for (true, hallucinated) pairs
    5. Compute ICFS v2 scores and select invariant features
    6. Train MuSAE-Inv probe (domain-augmented L1-LR)
    7. Run all baselines (Random, SelfCheckGPT-NLI proxy, SAPLMA, etc.)
    8. Compute TDV zero-shot and Logit Lens
    9. Evaluate all methods on all domains
    10. Generate result tables and figures

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --icfs-top-k 512 --musae-C 0.3
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from musae_inv.config import Config
from musae_inv.utils.seed import set_seed
from musae_inv.utils.io import save_pickle, load_pickle
from musae_inv.data.datasets import load_all_datasets
from musae_inv.data.preprocessing import build_icfs_features, build_env_icfs_from_counterfactual
from musae_inv.features.extraction import extract_features
from musae_inv.features.counterfactual import extract_counterfactual_pairs
from musae_inv.features.icfs import compute_icfs_v2
from musae_inv.models.probes import (
    MuSAEInvProbe,
    ERMAblationProbe,
    SingleLayerProbe,
    ConcatPCAProbe,
)
from musae_inv.models.saplma import train_saplma_probe
from musae_inv.models.model_loader import load_gemma_model, load_gemma_scope_saes
from musae_inv.evaluation.metrics import ResultsTracker, compute_metrics
from musae_inv.evaluation.baselines import (
    sae_entropy_score,
    tdv_direction_vector,
    tdv_score,
)
from musae_inv.evaluation.logit_lens import logit_lens_commitment_score
from musae_inv.visualization.tables import generate_result_tables


def parse_args():
    parser = argparse.ArgumentParser(description="MuSAE-Inv Training Pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--icfs-top-k", type=int, default=128)
    parser.add_argument("--musae-C", type=float, default=0.3)
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-logit-lens", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Configuration ─────────────────────────────────────
    if args.config:
        cfg = Config.load(args.config)
    else:
        cfg = Config(
            seed=args.seed,
            device=args.device,
            icfs_top_k=args.icfs_top_k,
            musae_C=args.musae_C,
            output_dir=args.output_dir,
        )
    cfg.make_dirs()
    set_seed(cfg.seed)

    print("=" * 70)
    print("  MuSAE-Inv Training Pipeline")
    print("=" * 70)
    print(f"  Device: {cfg.device}")
    print(f"  ICFS top-k: {cfg.icfs_top_k}")
    print(f"  MuSAE C: {cfg.musae_C}")
    print(f"  Output: {cfg.output_dir}")
    print("=" * 70)

    t_start = time.time()

    # ── Step 1: Load model and SAEs ───────────────────────
    print("\n[1/10] Loading Gemma-2-2B and Gemma Scope SAEs...")
    model, tokenizer = load_gemma_model(cfg)
    saes = load_gemma_scope_saes(cfg)

    # ── Step 2: Load datasets ─────────────────────────────
    print("\n[2/10] Loading datasets...")
    datasets = load_all_datasets(cfg)

    # ── Step 3: Extract SAE features ──────────────────────
    print("\n[3/10] Extracting SAE features...")
    feat_cache_path = cfg.features_dir / "feature_cache.pkl"

    if feat_cache_path.exists() and not cfg.force_recompute_feats:
        print("Loading cached features...")
        feature_cache = load_pickle(feat_cache_path)
    else:
        feature_cache = {}
        splits = [
            ("halueval_qa_train", datasets["halueval_qa_train"]),
            ("halueval_qa_val", datasets["halueval_qa_val"]),
            ("halueval_qa_test", datasets["halueval_qa_test"]),
            ("halueval_dial_test", datasets["halueval_dial_test"]),
            ("halueval_summ_test", datasets["halueval_summ_test"]),
            ("truthfulqa", datasets["truthfulqa"]),
        ]
        for name, df in splits:
            print(f"\n  {name} ({len(df)} examples)")
            feature_cache[name] = extract_features(
                df, model, tokenizer, saes, cfg.target_layers, cfg=cfg, desc=name,
            )
            save_pickle(feature_cache, feat_cache_path)
            if torch.cuda.is_available():
                print(f"  VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # ── Step 4: Counterfactual delta extraction ───────────
    print("\n[4/10] Extracting counterfactual deltas...")
    cf_cache_path = cfg.features_dir / "counterfactual_cache.pkl"

    if cf_cache_path.exists() and not cfg.force_recompute_feats:
        print("Loading cached counterfactual deltas...")
        cf_cache = load_pickle(cf_cache_path)
    else:
        cf_cache = {}
        for domain, key, n_pairs in [
            ("qa", "halueval_qa_pairs", cfg.n_cf_qa),
            ("dialogue", "halueval_dial_pairs", cfg.n_cf_dial),
            ("summarisation", "halueval_summ_pairs", cfg.n_cf_summ),
        ]:
            pairs = datasets[key].iloc[:n_pairs]
            print(f"\n  [{domain}] {len(pairs)} pairs")
            ft, ff, delta = extract_counterfactual_pairs(
                pairs, model, tokenizer, saes, cfg.target_layers, cfg=cfg, desc=domain,
            )
            cf_cache[domain] = {"feat_true": ft, "feat_false": ff, "delta": delta}
            del ft, ff
            gc.collect()
            torch.cuda.empty_cache()
        save_pickle(cf_cache, cf_cache_path)

    # ── Step 5: ICFS v2 scoring ───────────────────────────
    print("\n[5/10] Computing ICFS v2 scores...")
    icfs_cache_path = cfg.features_dir / "icfs_cache_v2.pkl"

    if icfs_cache_path.exists() and not cfg.force_recompute_icfs:
        icfs_data = load_pickle(icfs_cache_path)
        icfs_result = type("ICFSResult", (), icfs_data)()
    else:
        icfs_result = compute_icfs_v2(cf_cache, cfg.target_layers, cfg.icfs_top_k, cfg)
        save_pickle(
            {"scores": icfs_result.scores, "indices": icfs_result.indices},
            icfs_cache_path,
        )

    # Build ICFS feature matrices
    X_train = build_icfs_features(feature_cache["halueval_qa_train"], icfs_result.indices, cfg.icfs_top_k)
    X_val = build_icfs_features(feature_cache["halueval_qa_val"], icfs_result.indices, cfg.icfs_top_k)
    X_test = build_icfs_features(feature_cache["halueval_qa_test"], icfs_result.indices, cfg.icfs_top_k)
    X_dial = build_icfs_features(feature_cache["halueval_dial_test"], icfs_result.indices, cfg.icfs_top_k)
    X_summ = build_icfs_features(feature_cache["halueval_summ_test"], icfs_result.indices, cfg.icfs_top_k)
    X_tqa = build_icfs_features(feature_cache["truthfulqa"], icfs_result.indices, cfg.icfs_top_k)

    y_train = feature_cache["halueval_qa_train"]["labels"]
    y_val = feature_cache["halueval_qa_val"]["labels"]
    y_test = feature_cache["halueval_qa_test"]["labels"]
    y_dial = feature_cache["halueval_dial_test"]["labels"]
    y_summ = feature_cache["halueval_summ_test"]["labels"]
    y_tqa = feature_cache["truthfulqa"]["labels"]

    print(f"ICFS dim: {X_train.shape[1]} ({len(cfg.target_layers)}L × {cfg.icfs_top_k}feat)")

    # ── Step 6: Train MuSAE-Inv probe ────────────────────
    print("\n[6/10] Training MuSAE-Inv probe...")

    # Build dialogue training data from CF cache
    X_d0, y_d0 = build_env_icfs_from_counterfactual(cf_cache, "dialogue", 0, icfs_result.indices, cfg.icfs_top_k)
    X_d1, y_d1 = build_env_icfs_from_counterfactual(cf_cache, "dialogue", 1, icfs_result.indices, cfg.icfs_top_k)
    n_d = min(len(X_d0), len(X_d1), cfg.n_dial_train)
    X_dial_train = np.vstack([X_d0[:n_d], X_d1[:n_d]])
    y_dial_train = np.concatenate([np.zeros(n_d), np.ones(n_d)])

    probe = MuSAEInvProbe(cfg)
    probe.fit(X_train, y_train, X_dial_train, y_dial_train, X_val, y_val)

    # ── Step 7: ERM ablation ─────────────────────────────
    print("\n  Training ERM ablation (QA-only, L2)...")
    erm_probe = ERMAblationProbe(cfg)
    erm_probe.fit(X_train, y_train)

    # ── Step 8: Evaluate all methods ─────────────────────
    print("\n[7/10] Evaluating all methods...")
    tracker = ResultsTracker()

    # MuSAE-Inv
    print("\n── MuSAE-Inv (Ours) ──")
    for dom, X, y in [("QA", X_test, y_test), ("Dialogue", X_dial, y_dial),
                       ("Summ", X_summ, y_summ), ("TruthfulQA", X_tqa, y_tqa)]:
        tracker.register("MuSAE-Inv (Ours)", dom, y, probe.predict_proba(X))

    # MuSAE-ERM ablation
    print("\n── MuSAE-ERM (ablation) ──")
    for dom, X, y in [("QA", X_test, y_test), ("Dialogue", X_dial, y_dial),
                       ("Summ", X_summ, y_summ), ("TruthfulQA", X_tqa, y_tqa)]:
        tracker.register("MuSAE-ERM (no domain-aug, ablation)", dom, y, erm_probe.predict_proba(X))

    # TDV zero-shot
    print("\n── TDV Zero-Shot ──")
    tdv = tdv_direction_vector(probe.scaler.transform(X_train), y_train)
    for dom, X, y in [("QA", X_test, y_test), ("Dialogue", X_dial, y_dial),
                       ("Summ", X_summ, y_summ), ("TruthfulQA", X_tqa, y_tqa)]:
        scores = tdv_score(probe.scaler.transform(X), tdv)
        tracker.register("MuSAE + TDV (zero-shot)", dom, y, scores)

    np.save(cfg.results_dir / "tdv.npy", tdv)

    if not args.skip_baselines:
        # Random
        print("\n── Random ──")
        rng = np.random.RandomState(cfg.seed)
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
            sl_probe = SingleLayerProbe(l, cfg)
            sl_probe.fit(feature_cache["halueval_qa_train"]["sae_features"][l], y_train)
            for dom, fc_key, y in [("QA", "halueval_qa_test", y_test),
                                    ("Dialogue", "halueval_dial_test", y_dial),
                                    ("Summ", "halueval_summ_test", y_summ),
                                    ("TruthfulQA", "truthfulqa", y_tqa)]:
                p = sl_probe.predict_proba(feature_cache[fc_key]["sae_features"][l])
                tracker.register(f"SAE-L{l} LR", dom, y, p)

    # Logit Lens
    if not args.skip_logit_lens:
        print("\n[8/10] Computing Logit Lens...")
        ll_cache_path = cfg.results_dir / "logit_lens_scores.pkl"
        if ll_cache_path.exists():
            ll_cache = load_pickle(ll_cache_path)
        else:
            ll_cache = {}
            df_map = {"QA": "halueval_qa_test", "Dialogue": "halueval_dial_test",
                      "Summ": "halueval_summ_test", "TruthfulQA": "truthfulqa"}
            y_map = {"QA": y_test, "Dialogue": y_dial, "Summ": y_summ, "TruthfulQA": y_tqa}
            for dom in ["QA", "Dialogue", "Summ", "TruthfulQA"]:
                texts = datasets[df_map[dom]]["text"].astype(str).tolist()
                commit = logit_lens_commitment_score(
                    texts, model, tokenizer, cfg.target_layers,
                    max_length=128, device=cfg.device,
                )
                hall_score = 1.0 - commit
                ll_cache[dom] = (y_map[dom], hall_score, commit)
                gc.collect()
                torch.cuda.empty_cache()
            save_pickle(ll_cache, ll_cache_path)

        for dom, (y, hall_score, _) in ll_cache.items():
            tracker.register("Logit Lens (zero-shot)", dom, y, hall_score)

    # ── Step 9: Generate tables ──────────────────────────
    print("\n[9/10] Generating result tables...")
    generate_result_tables(tracker.results, cfg.results_dir)

    # Save config
    cfg.save(cfg.results_dir / "config.yaml")

    # ── Summary ──────────────────────────────────────────
    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"  MuSAE-Inv Training Complete — {elapsed / 60:.1f} minutes")
    print("=" * 70)
    print(f"  Results: {cfg.results_dir}")
    print(f"  Plots:   {cfg.plots_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
