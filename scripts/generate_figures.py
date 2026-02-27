#!/usr/bin/env python3
"""
Generate all 22 publication-quality figures from cached results.

Produces the full figure set for the MuSAE-Inv paper: heatmaps,
ablation curves, PCA/t-SNE embeddings, calibration plots, confusion
matrices, Sankey-style disagreement charts, etc.

Usage:
    python scripts/generate_figures.py --output-dir ./outputs
    python scripts/generate_figures.py --figures 1,3,5      # specific figures
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from musae_inv.config import Config
from musae_inv.utils.seed import set_seed
from musae_inv.utils.io import load_pickle
from musae_inv.visualization.plots import generate_all_figures


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Paper Figures")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--format", type=str, default="pdf", choices=["pdf", "png", "svg"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--figures", type=str, default="all",
                        help="Comma-separated figure numbers (1-22) or 'all'")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.load(args.config) if args.config else Config(output_dir=args.output_dir)
    set_seed(cfg.seed)

    print("=" * 60)
    print("  MuSAE-Inv Figure Generation")
    print("=" * 60)

    # Load feature cache
    feature_cache = load_pickle(cfg.features_dir / "feature_cache.pkl")

    # Load ICFS results
    icfs_data = load_pickle(cfg.features_dir / "icfs_cache_v2.pkl")

    # Figure selection
    if args.figures == "all":
        fig_list = list(range(1, 23))
    else:
        fig_list = [int(f) for f in args.figures.split(",")]

    print(f"\n  Generating {len(fig_list)} figures (format={args.format}, dpi={args.dpi})")

    generate_all_figures(
        feature_cache=feature_cache,
        icfs_data=icfs_data,
        cfg=cfg,
        fig_list=fig_list,
        fmt=args.format,
        dpi=args.dpi,
    )

    print(f"\n✅ Figures saved to {cfg.plots_dir}/")


if __name__ == "__main__":
    main()
