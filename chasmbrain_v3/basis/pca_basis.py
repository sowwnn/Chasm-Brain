"""PCA basis baseline (spec §4): geometry frequency vs data variance."""
import numpy as np

from .nested import NestedBasis


class PCABasis(NestedBasis):
    """SVD of X_train [n_images, n_voxels], same interface as LaplacianBasis.
    k(t) = round(t*K); project = component scores; lift = reconstruct from the first k(t) components."""

    def __init__(self, X_train: np.ndarray, K: int):
        mu = X_train.mean(axis=0)
        Xc = X_train - mu
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.mean = mu.astype(np.float64)
        self.components = Vt[:K].astype(np.float64)   # [K, n_voxels], rows are principal axes (unit norm)
        self.explained_variance = (S[:K] ** 2 / (len(X_train) - 1)).astype(np.float64)
        self.K = self.components.shape[0]

    def k(self, t: float) -> int:
        return max(1, round(t * self.K)) if t > 0 else 0

    def project(self, v: np.ndarray, t: float) -> np.ndarray:
        kt = self.k(t)
        return self.components[:kt] @ (v - self.mean)

    def lift(self, c: np.ndarray, t: float) -> np.ndarray:
        kt = self.k(t)
        return self.components[:kt].T @ c[:kt] + self.mean

    def band(self, t_lo: float, t_hi: float) -> np.ndarray:
        return np.arange(self.k(t_lo), self.k(t_hi))
