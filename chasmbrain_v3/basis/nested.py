"""Nested basis interface (spec §4). V_1 ⊂ V_2 ⊂ ... ⊂ V_m = R^n; the model calls only these methods.

Invariants (tested in tests/test_nested_basis.py):
  1. project(lift(c, t), t) == c
  2. lift at t_hi using only the coefficients of t_lo equals lift at t_lo
"""
import numpy as np


class NestedBasis:
    def k(self, t: float) -> int:
        """Number of dimensions at level t."""
        raise NotImplementedError

    def project(self, v: np.ndarray, t: float) -> np.ndarray:
        """Coefficients of v in V_t, shape [k(t)]."""
        raise NotImplementedError

    def lift(self, c: np.ndarray, t: float) -> np.ndarray:
        """Reconstruct voxels from coefficients (Moore–Penrose inverse), shape [n_voxels]."""
        raise NotImplementedError

    def band(self, t_lo: float, t_hi: float) -> np.ndarray:
        """Indices of the dimensions added when going from t_lo to t_hi."""
        raise NotImplementedError
