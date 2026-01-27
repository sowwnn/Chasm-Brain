
import torch
import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from ARMNet.dual_stream_cnn_model import DualStreamCNN

def benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Tiny Config (Current)
    model = DualStreamCNN(
        vis_dim=768,
        compact_shape=(61, 46, 42),
        n_voxels=15724,
        base_channels_3d=16,
        base_channels_1d=32,
        depths=[1, 1, 1, 1],
        fusion_type='mlp',
        fusion_dim=64,
        use_coarse_supervision=True
    ).to(device)
    
    B = 4
    vis_feat = torch.randn(B, 768).to(device)
    mask = torch.zeros((61, 46, 42)).to(device)
    coords = torch.randint(0, 40, (15724, 3)).to(device)
    
    # Warmup
    print("Warming up...")
    for _ in range(5):
        _ = model(vis_feat, mask, coords)
        
    torch.cuda.synchronize()
    
    # Benchmark
    print("Benchmarking Model Forward Pass...")
    t0 = time.time()
    n_iters = 50
    for _ in range(n_iters):
        _ = model(vis_feat, mask, coords)
    torch.cuda.synchronize()
    dt = time.time() - t0
    
    print(f"Average Forward Time: {dt/n_iters*1000:.2f} ms/batch")
    print(f"Estimated Iterations per second: {n_iters/dt:.2f} it/s")

if __name__ == "__main__":
    benchmark()
