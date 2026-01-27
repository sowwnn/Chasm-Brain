import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return x + self.block(x)

class ARMNet(nn.Module):
    def __init__(self, input_dim=768, output_dim=15724, hidden_dim=1024, parcel_dim=1000, dropout=0.3):
        super().__init__()
        
        # 1. Feature Extractor (Nâng tầm visual features)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Các khối Residual để học sâu hơn về đặc trưng hình ảnh
        self.deep_features = nn.Sequential(
            ResidualBlock(hidden_dim, dropout),
            ResidualBlock(hidden_dim, dropout)
        )

        # 2. Parcel-level Guidance (Nhánh học cấu trúc vùng não)
        # Giả định 1000 parcelation là một bước đệm quan trọng
        self.parcel_bottleneck = nn.Sequential(
            nn.Linear(hidden_dim, parcel_dim),
            nn.LayerNorm(parcel_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 3. Residual Prediction (Voxel level)
        # Đi từ Parcel-level lên Voxel-level giúp mô hình ổn định hơn
        self.res_head = nn.Sequential(
            nn.Linear(parcel_dim, parcel_dim * 2),
            nn.GELU(),
            nn.Linear(parcel_dim * 2, output_dim)
        )

        # 4. Gating Network (Cải tiến với Attention-like mechanism)
        self.gate_head = nn.Sequential(
            nn.Linear(hidden_dim, parcel_dim),
            nn.GELU(),
            nn.Linear(parcel_dim, output_dim),
            nn.Sigmoid()
        )

        # 5. Global Scale (Độ hưng phấn chung của não bộ)
        self.scale_net = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Softplus()
        )

    def forward(self, x, mean_fmri):
        """
        x: [B, 768] (DINOv2)
        mean_fmri: [B, 15724]
        """
        # Trích xuất đặc trưng sâu
        latent = self.input_proj(x)
        latent = self.deep_features(latent)
        
        # Học ở mức độ Parcel (vùng não) trước
        parcels = self.parcel_bottleneck(latent)
        
        # Dự đoán chi tiết mức Voxel
        res = self.res_head(parcels)
        gate = self.gate_head(latent)
        scale = self.scale_net(latent)
        
        # Modulation: Kết hợp thông tin
        # Tín hiệu = Mean + Scale * (Chi tiết * Vùng hoạt động)
        out = mean_fmri + (scale * res * gate)
        
        return out