"""Spatially constrained hierarchical clustering on train responses (spec §4)."""
import numpy as np
from .nested import NestedBasis


def fit_tree(X_train: np.ndarray, coords: np.ndarray, A, linkage: str = "ward", spatial_weight: float = 0.0):
    """sklearn AgglomerativeClustering(connectivity=A) on train response correlations."""
    raise NotImplementedError


def cut(tree, n_clusters: int) -> np.ndarray:
    raise NotImplementedError


class ClusterBasis(NestedBasis):
    """k(t) = nearest level; project = cluster mean; lift = broadcast cluster value. Nested by the tree."""

    def __init__(self, tree, levels=(300, 1000, 5000, 15000)):
        self.tree, self.levels = tree, list(levels)
