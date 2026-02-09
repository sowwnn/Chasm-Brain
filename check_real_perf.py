
import torch
import yaml
import numpy as np
import sys
from pathlib import Path

# Add workspace to path
sys.path.append('/home/sowwn/Workspace/ws/2026/I2fMRI')

from ARMNet import DenoisingARM
from data.neuroflux_dataset import create_dataloaders

def check_real_performance():
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
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # Load data
    _, val_loader = create_dataloaders(
        datalist_path=config['data']['datalist_path'],
        fmri_path=config['data']['fmri_path'],
        train_embeddings_path=config['data']['train_embeddings_path'],
        test_embeddings_path=config['data']['test_embeddings_path'],
        subjects=[config['data']['subject']],
        batch_size=32,
        parcel_labels_path=None,
        average_trials_train=config['data']['average_trials_train'],
        average_trials_val=config['data']['average_trials_val']
    )
    
    batch = next(iter(val_loader))
    target = batch['fmri'].to(device)
    vis_feat = batch['embedding'].to(device)
    
    print(f"Target stats - Mean: {target.mean().item():.4f}, Std: {target.std().item():.4f}")
    
    # Súng mean = 0 (giả sử dữ liệu đã được center)
    zero_mean = torch.zeros_like(target[0])
    batch_mean_zero = zero_mean.unsqueeze(0).expand(target.size(0), -1)
    
    # Test với Pure Noise (Strategy dùng trong training script)
    x_input = torch.randn_like(target)
    with torch.no_grad():
        pred = model(x_input, vis_feat, batch_mean_zero)
        
        pearsons = []
        for i in range(pred.size(0)):
            p = np.corrcoef(pred[i].cpu().numpy(), target[i].cpu().numpy())[0, 1]
            pearsons.append(p)
        
        print(f"\nReal Validation Pearson (with zero mean fallback): {np.mean(pearsons):.4f}")
        print(f"Prediction stats - Mean: {pred.mean().item():.4f}, Std: {pred.std().item():.4f}")

if __name__ == '__main__':
    check_real_performance()
