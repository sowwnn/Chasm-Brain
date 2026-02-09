import torch
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device).eval()

print("Common attributes/methods in DinoVisionTransformer:")
attrs = dir(model)
for attr in attrs:
    if "atten" in attr.lower() or "self" in attr.lower():
        print(f" - {attr}")

# Try to see if it's DinoVisionTransformer or something else
print(f"Model class: {type(model)}")
