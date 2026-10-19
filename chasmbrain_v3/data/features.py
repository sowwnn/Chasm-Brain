"""Features for NSD images (spec §3). DINOv2 ViT-B/14 is already cached in dataset/huwng/ (loaded via
nsd_loader.load_dinov2). CLIP ViT-B/16 and ResNet50 are not available pre-extracted and are only needed
for E6 (spec §10); they are left unimplemented until that experiment, extracted from BOLD5000's stimuli
the same way dataset/BOLD5000/processed/CSI1 was built."""
BACKBONES = ("dinov2_vitb14", "clip_vitb16", "resnet50")


def extract(backbone: str, images, layer: int):
    if backbone == "dinov2_vitb14":
        raise NotImplementedError("DINOv2 features for NSD are pre-cached; use nsd_loader.load_dinov2 instead")
    raise NotImplementedError(f"{backbone}: needed only for E6 (spec §10), not yet implemented")
