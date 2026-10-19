"""
Visualize Early vs Higher Visual Areas in 3D Brain Viewer

Color scheme:
- Blue: Early visual cortex (V1, V2, V3 - processes spatial/low-level info from patches)
- Red: Higher visual areas (processes object identity from CLS token)

Usage:
    python scripts/visualize_what_where_3d.py --port 8050
"""

import argparse
import numpy as np
import nibabel as nib
import cortex


def load_streams_data():
    """Load NSD streams (visual areas) data."""
    print("Loading streams data...")

    nsd_roi_path = '/mnt/HDD/dataset/NSD_raw/nsddata/ppdata/subj01/func1pt8mm/roi/'
    streams_nii = nib.load(nsd_roi_path + 'streams.nii.gz')
    streams_data = streams_nii.get_fdata()

    print(f"  Shape: {streams_data.shape}")
    print(f"  Unique values: {np.unique(streams_data).astype(int).tolist()}")

    return streams_data


def create_region_volume(streams_data):
    """Create volume with Early (blue) vs Higher (red) visual areas.

    Stream IDs in NSD:
        1: Early (V1, V2, V3)
        2: Midventral
        3: Midlateral
        4: Lateral
        5: Ventral
        6: Parietal
    """
    print("Creating region volume...")

    EARLY_STREAM_ID = 1

    # Create volume: Early = -1 (blue), Others = +1 (red)
    region_volume = np.zeros_like(streams_data, dtype=np.float32)

    # Early visual cortex -> Blue (negative)
    early_mask = streams_data == EARLY_STREAM_ID
    region_volume[early_mask] = -1.0

    # Higher visual areas -> Red (positive)
    other_mask = (streams_data > 0) & (streams_data != EARLY_STREAM_ID)
    region_volume[other_mask] = 1.0

    print(f"  Early (Blue): {early_mask.sum()} voxels")
    print(f"  Higher (Red): {other_mask.sum()} voxels")

    return region_volume


def launch_viewer(region_volume, port=8050):
    """Launch Pycortex 3D web viewer."""
    print(f"Launching 3D viewer on port {port}...")

    SUBJECT = 'subj01'
    TRANSFORM = 'func'

    # Create pycortex volume
    # Note: pycortex reverses shape, so we need .T
    # RdBu_r: Blue (negative) -> White (zero) -> Red (positive)
    region_vol = cortex.Volume(
        region_volume.T.astype(np.float64),
        subject=SUBJECT,
        xfmname=TRANSFORM,
        cmap='RdBu_r',
        vmin=-1.0,
        vmax=1.0
    )

    dataset = cortex.Dataset(early_vs_higher=region_vol)

    print("\n" + "=" * 60)
    print("EARLY vs HIGHER VISUAL AREAS")
    print("=" * 60)
    print("  Blue: Early visual cortex (V1, V2, V3)")
    print("  Red:  Higher visual areas (ventral/dorsal streams)")
    print(f"\nServer: http://localhost:{port}")
    print("Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    cortex.webgl.show(dataset, open_browser=True, port=port)


def main():
    parser = argparse.ArgumentParser(description='Visualize Early vs Higher visual areas')
    parser.add_argument('--port', type=int, default=8050, help='Port for web viewer')
    args = parser.parse_args()

    streams_data = load_streams_data()
    region_volume = create_region_volume(streams_data)
    launch_viewer(region_volume, args.port)


if __name__ == '__main__':
    main()
