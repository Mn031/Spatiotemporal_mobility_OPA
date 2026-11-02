"""
OPA Package for Spatiotemporal Mobility Trajectories.
One-Shot Private Aggregation (CRYPTO 2025) and Privacy Evaluation.
"""

from .crypto import (
    LWR_PRG,
    share,
    reconstruct,
    quantize_to_zp,
    dequantize_from_zp,
    DEFAULT_Q,
    DEFAULT_P,
    DEFAULT_SCALE,
)
from .model import NextLocationModel
from .dataset import (
    CANARY_HEX_SEQUENCE,
    CANARY_STRING,
    CANARY_START_HEX,
    LocationTokenizer,
    get_client_dataloaders,
    generate_synthetic_gps_data,
    discretize_gps_trajectories,
)
from .opa_flwr import OPAClient, OPAFedAvg, OPAStrategy
from .attack import (
    compute_sequence_perplexity,
    greedy_rollout_trajectory,
    tab_attack_leakage_analysis,
    get_quantitative_privacy_metrics,
    print_quantitative_report,
)

__all__ = [
    "LWR_PRG",
    "share",
    "reconstruct",
    "quantize_to_zp",
    "dequantize_from_zp",
    "DEFAULT_Q",
    "DEFAULT_P",
    "DEFAULT_SCALE",
    "NextLocationModel",
    "CANARY_HEX_SEQUENCE",
    "CANARY_STRING",
    "CANARY_START_HEX",
    "LocationTokenizer",
    "get_client_dataloaders",
    "generate_synthetic_gps_data",
    "discretize_gps_trajectories",
    "OPAClient",
    "OPAFedAvg",
    "OPAStrategy",
    "compute_sequence_perplexity",
    "greedy_rollout_trajectory",
    "tab_attack_leakage_analysis",
    "get_quantitative_privacy_metrics",
    "print_quantitative_report",
]

