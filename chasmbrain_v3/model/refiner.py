"""t-conditioned Refiner (spec §5). Predicts only the new band; never re-emits old coefficients.

Append-only constraint is structural: step() concatenates c_prev untouched (tested in tests/test_refiner_append_only.py).
Budget: d_hidden=512, encoder ~2M, refiner ~5-8M, total < 15M (TBEn is 17.7M).
"""
import math

import torch
import torch.nn as nn


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """t: [B] float in [0,1] -> [B, dim]. Standard transformer-style sin/cos embedding."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.shape[-1] < dim:
        emb = torch.cat([emb, torch.zeros(*emb.shape[:-1], dim - emb.shape[-1], device=t.device, dtype=t.dtype)], dim=-1)
    return emb


class Refiner(nn.Module):
    def __init__(self, encoder, basis, d_hidden: int = 512, t_embed_dim: int = 64):
        super().__init__()
        self.encoder, self.basis = encoder, basis
        self.d_hidden, self.t_embed_dim = d_hidden, t_embed_dim
        K = basis.K   # total basis dimensions; the head always emits all K, forward() slices to the band

        self.t_mlp = nn.Sequential(nn.Linear(2 * t_embed_dim, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_hidden))
        self.c_prev_proj = nn.Linear(K, d_hidden)   # proj(c_prev): c_prev zero-padded to K before this layer
        self.z_mlp = nn.Sequential(
            nn.Linear(3 * d_hidden, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_hidden), nn.GELU(),
        )
        # Shared head across every basis dimension, modulated by a learned per-dimension band embedding.
        # A plain z + band_embed[j] -> Linear(d_hidden, 1) collapses to (f(z) + g(band_embed[j])), i.e. a
        # per-sample constant plus a per-dimension constant -- it cannot represent any real interaction
        # between the sample and the target dimension, so it cannot fit arbitrary per-dimension targets
        # (verified empirically: such a head fails to overfit even a single random batch). Concatenating
        # z and band_embed[j] before a small MLP lets the head's weights multiply the two together, which
        # is the minimum needed for per-dimension predictions that actually depend on the input.
        self.head = nn.Sequential(nn.Linear(2 * d_hidden, d_hidden), nn.GELU(), nn.Linear(d_hidden, 1))
        self.band_embed = nn.Parameter(torch.randn(K, d_hidden) * 0.02)   # learned per-dimension embedding

    def compute_z(self, cls, patch, c_prev, t_prev: float, t: float):
        """Shared trunk: h = encoder; e_t = sinusoidal(t) (+) sinusoidal(t_prev) -> MLP;
        z = MLP([h_fused; e_t; proj(c_prev)]). Used by forward() and by the HaltHead at inference
        (run_dynamic, spec §7) so the two never compute z differently."""
        B = cls.shape[0]
        h_cls, h_patch, _ = self.encoder(cls, patch)
        h_fused = h_cls + h_patch   # FusionGate is a separate analysis path (spec §5), not used in the main forward

        t_prev_t = torch.full((B,), t_prev, device=cls.device, dtype=cls.dtype)
        t_t = torch.full((B,), t, device=cls.device, dtype=cls.dtype)
        e_t = self.t_mlp(torch.cat([sinusoidal_embedding(t_t, self.t_embed_dim),
                                     sinusoidal_embedding(t_prev_t, self.t_embed_dim)], dim=-1))

        K = self.basis.K
        c_prev_padded = torch.zeros(B, K, device=cls.device, dtype=cls.dtype)
        c_prev_padded[:, : c_prev.shape[1]] = c_prev
        c_proj = self.c_prev_proj(c_prev_padded)

        return self.z_mlp(torch.cat([h_fused, e_t, c_proj], dim=-1))   # [B, d_hidden]

    def forward(self, cls, patch, c_prev, t_prev: float, t: float):
        """-> c_new [B, |band(t_prev, t)|]. head([z, band_embed[j]]) -> c_new[j], one shared MLP applied
        per new dimension j, conditioned on that dimension's learned band embedding (spec §5: "a linear
        chung cho mọi chiều, cộng một band embedding học được cho từng chiều")."""
        z = self.compute_z(cls, patch, c_prev, t_prev, t)
        band = self.basis.band(t_prev, t)                            # indices into [0, K)
        n_band = len(band)
        z_rep = z[:, None, :].expand(-1, n_band, -1)                            # [B, |band|, d_hidden]
        embed_rep = self.band_embed[band][None, :, :].expand(z.shape[0], -1, -1)  # [B, |band|, d_hidden]
        c_new = self.head(torch.cat([z_rep, embed_rep], dim=-1)).squeeze(-1)    # shared MLP -> [B, |band|]
        return c_new

    def step(self, cls, patch, c_prev, t_prev: float, t: float):
        """-> c_full [B, k(t)] = concat(c_prev, forward(...)). c_prev passes through no layer."""
        c_new = self.forward(cls, patch, c_prev, t_prev, t)
        return torch.cat([c_prev, c_new], dim=1)

    def predict_voxels(self, c_full, t: float):
        """-> v_hat [B, n_voxels] = basis.lift(c_full, t), applied per batch row."""
        return torch.stack([torch.as_tensor(self.basis.lift(c_full[i].detach().cpu().numpy(), t),
                                             device=c_full.device, dtype=c_full.dtype) for i in range(c_full.shape[0])])


class HaltHead(nn.Module):
    """Dynamic-step head (spec §7), trained in a second phase on the frozen Refiner. Optional."""

    def __init__(self, d_hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_hidden, d_hidden), nn.GELU(), nn.Linear(d_hidden, 1))

    def forward(self, z):
        """-> p_halt [B, 1], sigmoid."""
        return torch.sigmoid(self.net(z))
