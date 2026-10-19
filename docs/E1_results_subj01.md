<!-- Source: chasmbrain_v3_artifacts/experiments/e3_steps_ablation/subj01_summary.json, run 2026-10-20. -->

# E1 — n_steps ablation (subj01)

Spec §10, E1: "Multi-step có thắng single-shot cùng basis?" Gate: Pearson n_steps > 1 cao hơn
n_steps = 1 có ý nghĩa (bootstrap CI không chồng).

## Setup

- Subject 1, Laplacian basis (K=512, 26-connectivity, per-component eigenmodes), d_hidden=256,
  t_embed_dim=64.
- Full spec-compliant training: 8000 train images, 1000 val, 1000 test (from `dataset/huwng`).
- AdamW lr=3e-4, batch 64, early stopping patience=15, max 150 epochs.
- Bootstrap CI: 2000 resamples, 2000-voxel subsample, GPU-vectorized.
- `python -m chasmbrain_v3.experiments.e3_steps_ablation --subj 1 --steps 1 2 4 8 --max_epochs 150 --patience 15 --d_hidden 256 --t_embed_dim 64 --K 512`

## Results

| n_steps | schedule | best val epoch | val Pearson | test Pearson | test Pearson 95% CI |
|---|---|---|---|---|---|
| 1 | [1.0] | 90 | 0.2601 | **0.2578** | [0.2484, 0.2677] |
| 2 | [0.02, 1.0] | 34 | 0.1725 | 0.1720 | [0.1597, 0.1846] |
| 4 | [0.02, 0.07, 0.27, 1.0] | 149 (max) | 0.2413 | 0.2398 | [0.2296, 0.2502] |
| 8 | [0.02 ... 0.57, 1.0] | 145 | 0.2389 | 0.2315 | [0.2211, 0.2423] |

## Gate: **FAIL**

No multi-step config beats single-shot with non-overlapping bootstrap CI.

- n_steps=2 is clearly worse (CI well below n_steps=1, no overlap).
- n_steps=4 is close but its CI upper bound (0.250) sits below n_steps=1's lower bound (0.248)
  only marginally -- the two intervals barely fail to overlap in n_steps=1's favor, i.e.
  single-shot is still ahead, not behind.
- n_steps=8 is further below n_steps=4, suggesting more steps does not monotonically help once
  the step count gets large relative to this basis/budget.
- n_steps=4 and n_steps=8 had not converged to a stable plateau by their early-stop point /
  max_epochs (n_steps=4 was still improving at epoch 149, i.e. hit the epoch cap rather than
  plateauing) -- more epochs might close the gap further, but the gate as specified in the spec
  is already decided on the numbers above.

## What this means (spec §10 decision rule)

Per spec: "Nếu rớt → Bỏ chữ multi-step; bài thành 'spectral basis cho encoding'; cân nhắc lại."

This is a real negative result on real NSD data, not an infrastructure or code issue (pipeline
verified end-to-end: bug-fixed Refiner head, confirmed correct gradient flow, confirmed stable
training on held-out loss). Before reframing the paper, worth considering:

1. Whether n_steps=4 training to convergence (more epochs, since it was still climbing at the
   cap) changes the comparison -- it is the closest config and did not plateau.
2. Whether the same ablation on a different basis (cluster tree, spec §4 pilot option) gives a
   different answer before concluding this is basis-independent.
3. Whether E2 (Laplacian vs PCA vs cluster tree vs smoothed ridge) should still run regardless,
   since it answers a different question (is the basis itself useful) independent of the
   multi-step result.

E0 → E1 → E2 are gates in sequence per spec; E1 failing means **E3 should not run** until this is
resolved, per the spec's stated stopping rule.
