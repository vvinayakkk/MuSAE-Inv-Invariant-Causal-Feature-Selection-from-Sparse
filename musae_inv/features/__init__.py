"""Features subpackage: extraction, counterfactual deltas, and ICFS scoring."""

from musae_inv.features.extraction import extract_features, get_hidden_states, hidden_to_sae
from musae_inv.features.counterfactual import extract_counterfactual_pairs
from musae_inv.features.icfs import compute_icfs_v2, ICFSResult
