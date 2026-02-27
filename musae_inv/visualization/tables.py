"""
Result table generation for the MuSAE-Inv paper.

Creates publication-quality CSV and LaTeX tables for:
- Main AUROC/BalAcc/AUPRC/Brier results across all methods and domains
- Cross-domain drop analysis
- Ablation study results
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from musae_inv.config import Config


METHODS_ORDER = [
    "Random",
    "SelfCheckGPT-NLI proxy",
    "Logit Lens (zero-shot)",
    "MuSAE + TDV (zero-shot)",
    "SAE-L6 LR",
    "SAE-L12 LR",
    "SAE-L18 LR",
    "SAE-L25 LR",
    "SAPLMA (SAE-L18 MLP)",
    "Concat-4L + PCA + LR",
    "MuSAE-ERM (no domain-aug, ablation)",
    "MuSAE-Inv (Ours)",
]

DOMAINS = ["QA", "Dialogue", "Summ", "TruthfulQA"]


def make_metric_table(
    all_results: Dict,
    metric: str = "auroc",
    methods: List[str] = None,
    domains: List[str] = None,
) -> pd.DataFrame:
    """Create a metric table across methods and domains.

    Parameters
    ----------
    all_results : dict
        Nested dict: method → domain → metrics.
    metric : str
        Metric name (auroc, auprc, balacc, brier).
    methods : list[str], optional
        Method order. Uses default if None.
    domains : list[str], optional
        Domain list. Uses default if None.

    Returns
    -------
    pd.DataFrame
        Table with Method, domain columns, and Avg.
    """
    if methods is None:
        methods = METHODS_ORDER
    if domains is None:
        domains = DOMAINS

    rows = []
    for m in methods:
        if m not in all_results:
            continue
        row = {"Method": m}
        vals = []
        for d in domains:
            v = all_results[m].get(d, {}).get(metric, float("nan"))
            row[d] = f"{v * 100:.2f}" if not np.isnan(v) else "—"
            if not np.isnan(v):
                vals.append(v)
        row["Avg"] = f"{np.mean(vals) * 100:.2f}" if vals else "—"
        rows.append(row)
    return pd.DataFrame(rows)


def make_drop_table(
    all_results: Dict,
    methods: List[str] = None,
) -> pd.DataFrame:
    """Create cross-domain AUROC drop table.

    Parameters
    ----------
    all_results : dict
        Results from all methods.
    methods : list[str], optional
        Method order.

    Returns
    -------
    pd.DataFrame
        Table with drop values in percentage points.
    """
    if methods is None:
        methods = METHODS_ORDER

    rows = []
    for m in methods:
        if m not in all_results:
            continue
        qa = all_results[m].get("QA", {}).get("auroc", np.nan)
        row = {"Method": m}
        for d in ["Dialogue", "Summ", "TruthfulQA"]:
            v = all_results[m].get(d, {}).get("auroc", np.nan)
            drop = (qa - v) * 100 if not (np.isnan(qa) or np.isnan(v)) else np.nan
            row[f"Drop to {d}"] = f"{drop:+.2f}pp" if not np.isnan(drop) else "—"
        rows.append(row)
    return pd.DataFrame(rows)


def generate_result_tables(
    all_results: Dict,
    output_dir: Path,
) -> Dict[str, pd.DataFrame]:
    """Generate and save all result tables.

    Parameters
    ----------
    all_results : dict
        Results from all methods.
    output_dir : Path
        Directory to save CSV files.

    Returns
    -------
    dict[str, pd.DataFrame]
        Generated tables keyed by name.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {}

    for name, metric in [("AUROC", "auroc"), ("BalAcc", "balacc"),
                          ("AUPRC", "auprc"), ("Brier", "brier")]:
        t = make_metric_table(all_results, metric)
        print(f"\nTABLE — {name}")
        print(t.to_string(index=False))
        t.to_csv(output_dir / f"table_{metric}.csv", index=False)
        tables[name] = t

    drop_table = make_drop_table(all_results)
    print("\nTABLE — QA→Domain AUROC Drop")
    print(drop_table.to_string(index=False))
    drop_table.to_csv(output_dir / "table_drop.csv", index=False)
    tables["Drop"] = drop_table

    print(f"\n✅ Tables saved to {output_dir}")
    return tables
