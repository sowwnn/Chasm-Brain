"""Statistics (spec §9). Every table: mean over 4 subjects ± SD, bootstrap CI, p vs TBEn."""
import numpy as np


def paired_subject_test(metric_A: np.ndarray, metric_B: np.ndarray) -> dict:
    """Paired t-test and Wilcoxon over 4 subjects."""
    raise NotImplementedError


def bootstrap_ci(v_pred: np.ndarray, v_true: np.ndarray, n: int = 10000) -> dict:
    """Bootstrap over images; 95% CI for Pearson and MSE."""
    raise NotImplementedError
