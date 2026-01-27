"""
2.5D fMRI Model: Xử lý volume 3D bằng 2D CNN trên từng lát + liên kết lát bằng 1D conv.
Input: [B, 1, D, H, W] (D: số lát, HxW: kích thước lát)
Output: [B, 1, D, H, W] (hoặc flatten về 1D)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FMRI2p5DNet(nn.Module):
    def __init__(self, depth=17, height=30, width=30, n_slices=17, n_channels=16):
        super().__init__()
        self.depth = depth
        self.height = height
        self.width = width
        self.n_slices = n_slices
        self.n_channels = n_channels
        # 2D CNN cho từng lát
        self.slice_encoder = nn.Sequential(
            nn.Conv2d(1, n_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_channels, n_channels, 3, padding=1),
            nn.ReLU()
        )
        # Liên kết lát bằng 1D conv
        self.slice_connector = nn.Conv1d(n_channels, n_channels, 3, padding=1)
        # Decoder 2D cho từng lát
        self.slice_decoder = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_channels, 1, 3, padding=1)
        )

    def forward(self, x):
        # x: [B, 1, D, H, W]
        B, C, D, H, W = x.shape
        # Xử lý từng lát: reshape về [B*D, 1, H, W]
        x_2d = x.permute(0,2,1,3,4).reshape(B*D, 1, H, W)
        feat_2d = self.slice_encoder(x_2d)  # [B*D, n_channels, H, W]
        # Global average pool từng lát: [B*D, n_channels]
        feat_flat = feat_2d.mean(dim=[2,3])
        # Kết nối lát: reshape về [B, D, n_channels] -> [B, n_channels, D]
        feat_seq = feat_flat.view(B, D, self.n_channels).transpose(1,2)
        feat_seq = self.slice_connector(feat_seq)  # [B, n_channels, D]
        # Trả lại từng lát: [B, D, n_channels]
        feat_seq = feat_seq.transpose(1,2).contiguous().view(B*D, self.n_channels, 1, 1)
        # Broadcast lại spatial: [B*D, n_channels, H, W]
        feat_2d = feat_2d + feat_seq.expand(-1, -1, H, W)
        # Decode từng lát
        out_2d = self.slice_decoder(feat_2d)  # [B*D, 1, H, W]
        # Ghép lại volume: [B, D, 1, H, W] -> [B, 1, D, H, W]
        out = out_2d.view(B, D, 1, H, W).permute(0,2,1,3,4)
        return out
