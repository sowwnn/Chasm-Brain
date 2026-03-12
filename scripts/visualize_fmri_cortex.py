"""
Visualize fMRI data on brain surface using Pycortex

This script provides utilities to visualize fMRI volumetric data
on the cortical surface using pycortex.

Prerequisites:
    1. Run setup_pycortex.py first to configure subject
    2. pip install pycortex nibabel numpy

Usage examples:
    # Basic visualization
    python scripts/visualize_fmri_cortex.py --nifti /path/to/data.nii.gz

    # With custom colormap and range
    python scripts/visualize_fmri_cortex.py --nifti /path/to/data.nii.gz --cmap hot --vmin 0 --vmax 5

    # Save static image instead of web viewer
    python scripts/visualize_fmri_cortex.py --nifti /path/to/data.nii.gz --save output.png
"""

import argparse
import numpy as np
import nibabel as nib
import cortex


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def load_nifti(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load NIfTI file and return data with affine.

    Args:
        filepath: Path to .nii or .nii.gz file

    Returns:
        data: 3D or 4D numpy array
        affine: 4x4 affine transformation matrix
    """
    img = nib.load(filepath)
    data = img.get_fdata()
    affine = img.affine
    print(f"Loaded: {filepath}")
    print(f"  Shape: {data.shape}")
    print(f"  Dtype: {data.dtype}")
    print(f"  Range: [{data.min():.3f}, {data.max():.3f}]")
    return data, affine


def create_volume(
    data: np.ndarray,
    subject: str = 'subj01',
    transform: str = 'func',
    cmap: str = 'viridis',
    vmin: float = None,
    vmax: float = None,
    description: str = ''
) -> cortex.Volume:
    """
    Create a pycortex Volume from 3D data.

    Args:
        data: 3D numpy array (x, y, z)
        subject: Pycortex subject name
        transform: Transform name (default: 'func')
        cmap: Matplotlib colormap name
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        description: Optional description for the volume

    Returns:
        cortex.Volume object

    Note:
        Pycortex expects data in (z, y, x) order, so we transpose.
    """
    # Auto compute vmin/vmax if not provided
    if vmin is None:
        vmin = float(np.nanmin(data[data != 0])) if np.any(data != 0) else 0
    if vmax is None:
        vmax = float(np.nanmax(data))

    # Pycortex expects transposed data
    data_t = data.T.astype(np.float64)

    vol = cortex.Volume(
        data_t,
        subject=subject,
        xfmname=transform,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        description=description
    )

    print(f"Created Volume:")
    print(f"  Subject: {subject}, Transform: {transform}")
    print(f"  Colormap: {cmap}, Range: [{vmin:.3f}, {vmax:.3f}]")

    return vol


def create_rgb_volume(
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
    subject: str = 'subj01',
    transform: str = 'func',
    alpha: np.ndarray = None
) -> cortex.VolumeRGB:
    """
    Create an RGB volume from separate channels.

    Args:
        red, green, blue: 3D arrays for each channel (0-255 or 0-1)
        subject: Pycortex subject name
        transform: Transform name
        alpha: Optional alpha channel

    Returns:
        cortex.VolumeRGB object
    """
    # Normalize to 0-255 if needed
    def normalize_channel(ch):
        ch = ch.T.astype(np.float64)
        if ch.max() <= 1.0:
            ch = ch * 255
        return np.clip(ch, 0, 255).astype(np.uint8)

    r = normalize_channel(red)
    g = normalize_channel(green)
    b = normalize_channel(blue)

    if alpha is not None:
        a = normalize_channel(alpha)
        return cortex.VolumeRGB(r, g, b, alpha=a, subject=subject, xfmname=transform)

    return cortex.VolumeRGB(r, g, b, subject=subject, xfmname=transform)


def create_2d_volume(
    data1: np.ndarray,
    data2: np.ndarray,
    subject: str = 'subj01',
    transform: str = 'func',
    cmap: str = 'RdBu_r',
    vmin: float = None,
    vmax: float = None,
    vmin2: float = None,
    vmax2: float = None
) -> cortex.Volume2D:
    """
    Create a 2D colormap volume (useful for showing two variables).

    Args:
        data1: First data dimension (mapped to hue)
        data2: Second data dimension (mapped to saturation/brightness)
        subject: Pycortex subject name
        transform: Transform name
        cmap: 2D colormap name
        vmin, vmax: Range for data1
        vmin2, vmax2: Range for data2

    Returns:
        cortex.Volume2D object
    """
    d1 = data1.T.astype(np.float64)
    d2 = data2.T.astype(np.float64)

    return cortex.Volume2D(
        d1, d2,
        subject=subject,
        xfmname=transform,
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        vmin2=vmin2, vmax2=vmax2
    )


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def show_webgl(
    dataset: cortex.Dataset,
    port: int = 8050,
    open_browser: bool = True
):
    """
    Launch interactive 3D web viewer.

    Args:
        dataset: Pycortex Dataset object
        port: Port for web server
        open_browser: Whether to open browser automatically
    """
    print(f"\nLaunching 3D viewer on http://localhost:{port}")
    print("Press Ctrl+C to stop")
    cortex.webgl.show(dataset, port=port, open_browser=open_browser)


def show_flatmap(
    volume: cortex.Volume,
    with_curvature: bool = True,
    with_rois: bool = False,
    pixelwise: bool = True,
    thick: float = 1,
    sampler: str = 'trilinear',
    height: int = 1024
) -> tuple:
    """
    Generate flatmap visualization.

    Args:
        volume: Pycortex Volume object
        with_curvature: Show curvature underlay
        with_rois: Show ROI boundaries
        pixelwise: Use pixel-wise mapping
        thick: Cortical thickness for sampling
        sampler: Interpolation method
        height: Output image height

    Returns:
        fig, ax: Matplotlib figure and axes
    """
    import matplotlib.pyplot as plt

    fig = cortex.quickflat.make_figure(
        volume,
        with_curvature=with_curvature,
        with_rois=with_rois,
        pixelwise=pixelwise,
        thick=thick,
        sampler=sampler,
        height=height
    )

    return fig


def show_inflated(
    volume: cortex.Volume,
    unfold: float = 0.5,
    height: int = 1024
):
    """
    Show data on inflated brain surface.

    Args:
        volume: Pycortex Volume object
        unfold: Inflation amount (0=folded, 1=flat)
        height: Output image height
    """
    import matplotlib.pyplot as plt

    fig = cortex.quickshow(
        volume,
        with_curvature=True,
        height=height
    )

    return fig


def save_static_image(
    volume: cortex.Volume,
    output_path: str,
    height: int = 2048,
    dpi: int = 300
):
    """
    Save flatmap as static image.

    Args:
        volume: Pycortex Volume object
        output_path: Output file path (.png, .svg, .pdf)
        height: Image height in pixels
        dpi: DPI for output
    """
    import matplotlib.pyplot as plt

    fig = cortex.quickflat.make_figure(
        volume,
        with_curvature=True,
        height=height
    )

    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_available_colormaps() -> list[str]:
    """Return list of commonly used colormaps for fMRI visualization."""
    return [
        # Sequential
        'viridis', 'plasma', 'inferno', 'magma', 'cividis',
        'hot', 'YlOrRd', 'OrRd', 'Reds', 'Blues',
        # Diverging (good for + vs - values)
        'RdBu_r', 'coolwarm', 'seismic', 'bwr',
        # Qualitative (good for discrete regions)
        'tab10', 'tab20', 'Set1', 'Set2', 'Set3',
        # Cyclic (good for phase data)
        'hsv', 'twilight', 'twilight_shifted'
    ]


def create_roi_volume(
    roi_data: np.ndarray,
    roi_ids: list[int] = None,
    subject: str = 'subj01',
    transform: str = 'func',
    cmap: str = 'tab20'
) -> cortex.Volume:
    """
    Create volume for ROI visualization.

    Args:
        roi_data: 3D array with integer ROI labels
        roi_ids: List of ROI IDs to include (None = all)
        subject: Pycortex subject name
        transform: Transform name
        cmap: Colormap for ROIs

    Returns:
        cortex.Volume object
    """
    if roi_ids is not None:
        mask = np.isin(roi_data, roi_ids)
        roi_masked = np.where(mask, roi_data, 0)
    else:
        roi_masked = roi_data

    unique_ids = np.unique(roi_masked[roi_masked > 0])
    print(f"ROI IDs: {unique_ids.tolist()}")

    return create_volume(
        roi_masked.astype(np.float32),
        subject=subject,
        transform=transform,
        cmap=cmap,
        vmin=0,
        vmax=unique_ids.max() if len(unique_ids) > 0 else 1
    )


def threshold_volume(
    data: np.ndarray,
    threshold: float,
    two_sided: bool = True
) -> np.ndarray:
    """
    Apply threshold to data (useful for statistical maps).

    Args:
        data: 3D data array
        threshold: Threshold value
        two_sided: If True, threshold both positive and negative values

    Returns:
        Thresholded data array
    """
    if two_sided:
        mask = np.abs(data) >= threshold
    else:
        mask = data >= threshold

    return np.where(mask, data, 0)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Visualize fMRI data on cortical surface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic visualization
    python %(prog)s --nifti data.nii.gz

    # Custom colormap and range
    python %(prog)s --nifti data.nii.gz --cmap hot --vmin 0 --vmax 5

    # Save as image
    python %(prog)s --nifti data.nii.gz --save output.png

    # Show on specific port
    python %(prog)s --nifti data.nii.gz --port 8888
        """
    )

    # Data input
    parser.add_argument('--nifti', type=str, required=True,
                        help='Path to NIfTI file (.nii or .nii.gz)')
    parser.add_argument('--volume-idx', type=int, default=None,
                        help='Volume index for 4D data (default: mean across time)')

    # Visualization options
    parser.add_argument('--subject', type=str, default='subj01',
                        help='Pycortex subject name')
    parser.add_argument('--transform', type=str, default='func',
                        help='Transform name')
    parser.add_argument('--cmap', type=str, default='viridis',
                        help='Colormap name')
    parser.add_argument('--vmin', type=float, default=None,
                        help='Minimum value for colormap')
    parser.add_argument('--vmax', type=float, default=None,
                        help='Maximum value for colormap')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Threshold value (values below are hidden)')

    # Output options
    parser.add_argument('--port', type=int, default=8050,
                        help='Port for web viewer')
    parser.add_argument('--save', type=str, default=None,
                        help='Save static image instead of web viewer')
    parser.add_argument('--no-browser', action='store_true',
                        help='Do not open browser automatically')

    args = parser.parse_args()

    # Load data
    data, _ = load_nifti(args.nifti)

    # Handle 4D data
    if data.ndim == 4:
        if args.volume_idx is not None:
            print(f"Using volume index: {args.volume_idx}")
            data = data[..., args.volume_idx]
        else:
            print("4D data detected, computing mean across time...")
            data = np.mean(data, axis=-1)

    # Apply threshold if specified
    if args.threshold is not None:
        print(f"Applying threshold: {args.threshold}")
        data = threshold_volume(data, args.threshold)

    # Create volume
    vol = create_volume(
        data,
        subject=args.subject,
        transform=args.transform,
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax
    )

    # Create dataset
    dataset = cortex.Dataset(data=vol)

    # Output
    if args.save:
        save_static_image(vol, args.save)
    else:
        show_webgl(dataset, port=args.port, open_browser=not args.no_browser)


if __name__ == '__main__':
    main()
