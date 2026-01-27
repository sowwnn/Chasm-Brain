# ARMNet Package
from .model import ARMNet
from .diffusion_model import DiffusionARM, DiffusionManager
from .diffusion_3d_model import DiffusionARM3D, DiffusionManager3D
from .cvae_model import HierarchicalCVAE
from .denoising_model import DenoisingARM
from .loss import PeakFocusedLoss, CVAELoss

# Optional: Graph models require torch_geometric
try:
    from .flow_graph_model import FlowGraphARM
    from .diffusion_graph_model import DiffusionGraphARM
    _has_graph = True
except ImportError:
    _has_graph = False
    FlowGraphARM = None
    DiffusionGraphARM = None

# Flow 3D Hybrid model (uses torch_geometric optionally)
from .flow_3d_hybrid_model import Flow3DHybridARM
from .flow_3d_loss import FlowMatching3DLoss, Metrics3D1D

__all__ = [
    'ARMNet',
    'DiffusionARM',
    'DiffusionManager',
    'DiffusionARM3D',
    'DiffusionManager3D',
    'HierarchicalCVAE',
    'DenoisingARM',
    'PeakFocusedLoss',
    'CVAELoss',
    'Flow3DHybridARM',
    'FlowMatching3DLoss',
    'Metrics3D1D',
]

if _has_graph:
    __all__.extend(['FlowGraphARM', 'DiffusionGraphARM'])
