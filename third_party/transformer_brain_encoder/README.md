# Vendored: transformer_brain_encoder (TBEn)

Source: https://github.com/Hosseinadeli/transformer_brain_encoder
Commit: `d4de87029ae490708461f464fb2650c0976c7aa2` (2026-04-09)
License: GPL-3.0 (see `LICENSE`)

Paper: Adeli, H., Sun, M., & Kriegeskorte, N. (2025). *Transformer brain
encoders explain human high-level visual responses.* arXiv:2505.17329.

## What is vendored

- `transformer.py` — verbatim from `models/transformer.py`. This is the DETR
  transformer with the authors' modifications, and it is the file that defines
  the TBEn readout: a decoder whose queries cross-attend to backbone tokens.
  Note that self-attention between queries is commented out by the authors, so
  each query attends only to the image tokens and to nothing else.
- `position_encoding.py` — from `models/position_encoding.py`. The only change
  is that `forward` takes `(x, mask)` directly instead of a `NestedTensor`,
  because we feed precomputed DINOv2 tokens rather than running their image
  backbone. The encoding math is untouched.
- `LICENSE` — GPL-3.0, as distributed by the authors.

## What is NOT vendored, and why

`models/brain_encoder.py`, `datasets/`, `engine.py`, and `main.py` are tied to
the Algonauts 2023 data layout (fsaverage surface vertices split into left and
right hemispheres, their own dataloaders and training loop). Our protocol uses a
single volumetric NSD mask, so those files cannot run unmodified. The readout
logic they contain is reimplemented against our voxel space in
`model/tben_official.py`, which documents each correspondence.

## Licensing note

GPL-3.0 is copyleft. Keeping these files in their own directory with the
original headers and license is the usual way to make provenance explicit, but
if this repository is released publicly, check how the GPL interacts with the
license you intend to use for your own code.
