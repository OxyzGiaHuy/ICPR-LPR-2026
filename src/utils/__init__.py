"""Utility functions."""
from src.utils.common import seed_everything
from src.utils.postprocess import decode_with_confidence
from src.utils.visualize import visualize_errors, analyze_confusion_pairs, print_error_analysis
from src.utils.character_balance import (
    compute_character_weights,
    weighted_ctc_loss,
    print_weight_statistics,
    compute_char_frequencies_from_dataset,
    CHAR_FREQUENCIES_FALLBACK
)
from src.utils.tta import predict_with_tta, validate_with_tta

__all__ = [
    "seed_everything",
    "decode_with_confidence",
    "visualize_errors",
    "analyze_confusion_pairs",
    "print_error_analysis",
    "compute_character_weights",
    "weighted_ctc_loss",
    "print_weight_statistics",
    "compute_char_frequencies_from_dataset",
    "CHAR_FREQUENCIES_FALLBACK",
    "predict_with_tta",
    "validate_with_tta",
]
