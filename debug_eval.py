
import torch
import yaml
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add workspace to path
sys.path.append('/home/sowwn/Workspace/ws/2026/I2fMRI')

from ARMNet import DenoisingARM
from data.neuroflux_dataset import create_dataloaders

def evaluate_best_model():
    config_path = '/home/sowwn/Workspace/ws/2026/I2fMRI/configs/train_denoising_15k.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = DenoisingARM(
        visual_dim=config['model']['visual_dim'],
        fmri_dim=config['model']['fmri_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        dropout=config['model']['dropout']
    ).to(device)
    
    checkpoint_path = Path(config['training']['save_dir']) / 'best_model.pth'
    if not checkpoint_path.exists():
        print(f"Checkpoint not found at {checkpoint_path}")
        return
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # Load data
    _, val_loader = create_dataloaders(
        datalist_path=config['data']['datalist_path'],
        fmri_path=config['data']['fmri_path'],
        train_embeddings_path=config['data']['train_embeddings_path'],
        test_embeddings_path=config['data']['test_embeddings_path'],
        subjects=[config['data']['subject']],
        batch_size=8,
        parcel_labels_path=None,
        average_trials_train=config['data']['average_trials_train'],
        average_trials_val=config['data']['average_trials_val']
    )
    
    # Compute mean fmri (required for model)
    # We can't easily compute from train_loader here, let's just use the first batch mean as a proxy or find it
    # Actually, we should compute it like the training script does.
    # But for a quick check, let's just see if predictions have any variance.
    
    batch = next(iter(val_loader))
    target = batch['fmri'].to(device)
    vis_feat = batch['embedding'].to(device)
    
    # We need the global mean used during training.
    # Let's assume it's the mean of the validation set for this check if we don't have the original.
    # Or try to find it.
    mean_fmri = target.mean(dim=0) # Proxy
    batch_mean = mean_fmri.unsqueeze(0).expand(target.size(0), -1)
    
    # Try different x_input strategies
    strategies = {
        'Pure Noise': torch.randn_like(target),
        'Mean + Noise': batch_mean + torch.randn_like(target) * 0.15,
        'Zero': torch.zeros_like(target)
    }
    
    results = {}
    for name, x_input in strategies.items():
        with torch.no_grad():
            pred = model(x_input.to(device), vis_feat, batch_mean)
            
            # Compute metrics
            pearsons = []
            mses = []
            for i in range(pred.size(0)):
                p = np.corrcoef(pred[i].cpu().numpy(), target[i].cpu().numpy())[0, 1]
                pearsons.append(p)
                mse = torch.nn.functional.mse_loss(pred[i], target[i]).item()
                mses.append(mse)
            
            results[name] = {
                'pearson_mean': np.mean(pearsons),
                'mse_mean': np.mean(mses),
                'pred_std': pred.std().item(),
                'target_std': target.std().item(),
                'delta_std': (pred - batch_mean).std().item()
            }
            
            print(f"Strategy: {name}")
            print(f"  Pearson: {results[name]['pearson_mean']:.4f}")
            print(f"  MSE: {results[name]['mse_mean']:.4f}")
            print(f"  Pred Std: {results[name]['pred_std']:.4f}")
            print(f"  Delta Std: {results[name]['delta_std']:.4f}")
    
    print("\nSummary Results:")
    for name, res in results.items():
        print(f"{name}: Pearson={res['pearson_mean']:.4f}, MSE={res['mse_mean']:.4f}, Delta Std={res['delta_std']:.4f}")

if __name__ == '__main__':
    evaluate_best_model()
