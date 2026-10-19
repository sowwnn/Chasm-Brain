# Legacy: CHASMBrain WACV round 2 code

Frozen code from the WACV submission and rebuttal. Tagged `wacv-round2-final`.
Not maintained; kept for reproducing old numbers and for reuse by `chasmbrain_v3/baselines/tben.py`.

Run from the repo root with both `legacy/` and the root on the path:

```bash
PYTHONPATH=legacy:. python legacy/scripts/train_hierarchical_v3.py --config legacy/configs/train_hierarchical_v3_30.yaml
```

Layout: `model/`, `data/`, `scripts/`, `configs/`, `results/`, `notebook/`, `manuscripts/`, `docs/` (figures).
`third_party/` stays at the repo root because TBEn imports it as `third_party.transformer_brain_encoder`.
Checkpoints that are still needed live in `checkpoints/` at the root (TBEn, BOLD5000, ChasmBrain reference runs);
everything else was copied to `/mnt/HDD/archive/I2fMRI-wacv/`.
