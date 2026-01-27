import torch
import yaml

from model import MyModel  # Giả sử bạn có một file model.py định nghĩa MyModel

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize model
model = MyModel(
    hidden_dim=config["model"]["hidden_dim"],
    num_layers=config["model"]["num_layers"],
    # ...other params...
)

# Load checkpoint if available
resume_path = config["model"].get("resume_path", "")
if resume_path:
    print(f"Resuming from checkpoint: {resume_path}")
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    # Nếu cần, load optimizer state, epoch, v.v.
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # start_epoch = checkpoint["epoch"] + 1
else:
    print("Training from scratch.")
    # start_epoch = 0

# ...existing training code...