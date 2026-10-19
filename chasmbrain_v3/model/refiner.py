"""t-conditioned Refiner (spec §5). Predicts only the new band; never re-emits old coefficients.

Append-only constraint is structural: step() concatenates c_prev untouched (tested in tests/test_refiner_append_only.py).
Budget: d_hidden=512, encoder ~2M, refiner ~5–8M, total < 15M (TBEn is 17.7M).
"""
import torch
import torch.nn as nn


class Refiner(nn.Module):
    def __init__(self, encoder, basis, d_hidden: int = 512, t_embed_dim: int = 64):
        super().__init__()
        self.encoder, self.basis = encoder, basis
        raise NotImplementedError

    def forward(self, cls, patch, c_prev, t_prev: float, t: float):
        """-> c_new [B, |band(t_prev, t)|]. h = encoder; e_t = sinusoidal(t) ⊕ sinusoidal(t_prev) -> MLP;
        z = MLP([h_fused; e_t; proj(c_prev)]); head(z) = shared linear + learned per-dimension band embedding."""
        raise NotImplementedError

    def step(self, cls, patch, c_prev, t_prev: float, t: float):
        """-> c_full [B, k(t)] = concat(c_prev, forward(...)). c_prev passes through no layer."""
        c_new = self.forward(cls, patch, c_prev, t_prev, t)
        return torch.cat([c_prev, c_new], dim=1)

    def predict_voxels(self, c_full, t: float):
        """-> v_hat [B, n_voxels] = basis.lift(c_full, t)."""
        raise NotImplementedError


class HaltHead(nn.Module):
    """Dynamic-step head (spec §7), trained in a second phase on the frozen Refiner. Optional."""

    def __init__(self, d_hidden: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, z):
        """-> p_halt [B, 1], sigmoid."""
        raise NotImplementedError
