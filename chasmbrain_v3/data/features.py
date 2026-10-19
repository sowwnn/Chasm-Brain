"""Frozen backbone features, cached to features/{backbone}_L{layer}_subj{N}.npy (spec §3)."""
BACKBONES = ("dinov2_vitb14", "clip_vitb16", "resnet50")


def extract(backbone: str, images, layer: int):
    """Return (cls [n, d], patch [n, 256, d])."""
    raise NotImplementedError
