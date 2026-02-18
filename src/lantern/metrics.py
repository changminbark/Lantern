from enum import Enum


class Metrics(Enum):
    """Enum of available training metrics for logging and reporting.

    Members:
        ACC: Classification accuracy.
        LOSS: Training/validation loss.
        F_ONE: F1 score (placeholder for future implementation).
    """

    ACC = "acc"
    LOSS = "loss"
    F1_MACRO = "f1_macro"


ALL_METRICS = list(Metrics)
