"""PCA basis baseline (spec §4): geometry frequency vs data variance."""
import numpy as np
from .nested import NestedBasis


class PCABasis(NestedBasis):
    """SVD of X_train, same interface as LaplacianBasis."""

    def __init__(self, X_train: np.ndarray, K: int):
        raise NotImplementedError
