"""Evaluation subpackage: metrics, baselines, and Logit Lens."""

from musae_inv.evaluation.metrics import compute_metrics, register_result
from musae_inv.evaluation.baselines import run_all_baselines
from musae_inv.evaluation.logit_lens import logit_lens_commitment_score
