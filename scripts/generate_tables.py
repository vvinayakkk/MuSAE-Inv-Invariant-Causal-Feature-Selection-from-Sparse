#!/usr/bin/env python3
"""
Generate all result tables (CSV format) from cached evaluation results.

Produces:
    - results_all.csv: Full results matrix (Method × Domain × Metric)
    - results_brier_drop.csv: ID-to-OOD Brier score degradation
    - results_auroc_drop.csv: ID-to-OOD AUROC drop
    - pairwise_comparison.csv: Statistical comparison vs MuSAE-Inv

Usage:
    python scripts/generate_tables.py --output-dir ./outputs
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from musae_inv.config import Config
from musae_inv.utils.io import load_pickle
from musae_inv.visualization.tables import generate_result_tables


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Result Tables")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.load(args.config) if args.config else Config(output_dir=args.output_dir)

    results_path = cfg.results_dir / "results_tracker.pkl"
    if results_path.exists():
        tracker = load_pickle(results_path)
        results = tracker if isinstance(tracker, dict) else tracker.results
    else:
        print("No cached results found. Run scripts/train.py first.")
        return

    generate_result_tables(results, cfg.results_dir)
    print(f"✅ Tables saved to {cfg.results_dir}/")


if __name__ == "__main__":
    main()
