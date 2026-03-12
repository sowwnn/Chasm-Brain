"""
Setup Pycortex with NSD FreeSurfer data (Manual import - no FreeSurfer required)

Run this ONCE after downloading NSD FreeSurfer data:
    python scripts/setup_pycortex.py
"""

import os
import json
import shutil
import numpy as np
import nibabel as nib
from nibabel import freesurfer as fs
import cortex


def read_freesurfer_surface(filepath):
    """Read FreeSurfer surface file."""
    coords, faces = fs.read_geometry(filepath)
    return coords, faces


def write_gifti_surface(filepath, coords, faces):
    """Write surface as GIFTI format for pycortex."""
    coord_array = nib.gifti.GiftiDataArray(
        data=coords.astype(np.float32),
        intent='NIFTI_INTENT_POINTSET',
        datatype='NIFTI_TYPE_FLOAT32'
    )
    face_array = nib.gifti.GiftiDataArray(
        data=faces.astype(np.int32),
        intent='NIFTI_INTENT_TRIANGLE',
        datatype='NIFTI_TYPE_INT32'
    )
    gii = nib.gifti.GiftiImage(darrays=[coord_array, face_array])
    nib.save(gii, filepath)


def setup_pycortex_manual(
    freesurfer_path: str,
    subject_name: str = 'subj01'
):
    """
    Manually import FreeSurfer surfaces into Pycortex without using FS tools.
    """
    print(f"Setting up Pycortex for {subject_name} (manual import)...")
    print(f"  FreeSurfer path: {freesurfer_path}")

    surf_dir = os.path.join(freesurfer_path, 'surf')

    # Get pycortex filestore path
    filestore = cortex.database.default_filestore
    subj_dir = os.path.join(filestore, subject_name)

    print(f"  Pycortex filestore: {filestore}")
    print(f"  Subject directory: {subj_dir}")

    # Create subject directories
    dirs_to_create = [
        'surfaces', 'transforms', 'cache', 'views', 'surface-info', 'anatomicals'
    ]
    for d in dirs_to_create:
        os.makedirs(os.path.join(subj_dir, d), exist_ok=True)

    # Surface mappings: freesurfer name -> pycortex name
    surface_types = {
        'white': 'wm',
        'pial': 'pia',
        'inflated': 'inflated',
        'sphere': 'sphere',
    }

    hemispheres = ['lh', 'rh']

    # Pycortex expects naming: {type}_{hemi}.gii (e.g., wm_lh.gii)
    print("\n  Importing surfaces...")
    for hemi in hemispheres:
        for fs_name, cx_name in surface_types.items():
            fs_file = os.path.join(surf_dir, f'{hemi}.{fs_name}')
            cx_file = os.path.join(subj_dir, 'surfaces', f'{cx_name}_{hemi}.gii')

            if os.path.exists(fs_file):
                try:
                    coords, faces = read_freesurfer_surface(fs_file)
                    write_gifti_surface(cx_file, coords, faces)
                    print(f"    {hemi}.{fs_name} -> {cx_name}_{hemi}.gii ({len(coords)} vertices)")
                except Exception as e:
                    print(f"    {hemi}.{fs_name}: ERROR - {e}")
            else:
                print(f"    {hemi}.{fs_name}: Not found, skipping")

    # Create fiducial surface (average of wm and pia)
    print("\n  Creating fiducial surfaces...")
    for hemi in hemispheres:
        wm_file = os.path.join(subj_dir, 'surfaces', f'wm_{hemi}.gii')
        pia_file = os.path.join(subj_dir, 'surfaces', f'pia_{hemi}.gii')
        fid_file = os.path.join(subj_dir, 'surfaces', f'fiducial_{hemi}.gii')

        if os.path.exists(wm_file) and os.path.exists(pia_file):
            wm_gii = nib.load(wm_file)
            pia_gii = nib.load(pia_file)

            wm_coords = wm_gii.darrays[0].data
            pia_coords = pia_gii.darrays[0].data
            faces = wm_gii.darrays[1].data

            fid_coords = (wm_coords + pia_coords) / 2
            write_gifti_surface(fid_file, fid_coords, faces)
            print(f"    fiducial_{hemi}.gii created")

    # Note: Flat maps require special FreeSurfer patch files (*.full.flat.patch.3d)
    # which have different format. Skipping flat map import - will use 3D viewer instead.
    print("\n  Flat maps: Skipped (NSD uses patch format, 3D viewer will be used instead)")

    # Import anatomical if available
    print("\n  Looking for anatomical...")
    mri_dir = os.path.join(freesurfer_path, 'mri')
    anat_files = ['T1.mgz', 'brain.mgz', 'orig.mgz']
    anat_imported = False

    for anat_name in anat_files:
        anat_src = os.path.join(mri_dir, anat_name)
        if os.path.exists(anat_src):
            try:
                img = nib.load(anat_src)
                anat_dst = os.path.join(subj_dir, 'anatomicals', 'raw.nii.gz')
                nib.save(nib.Nifti1Image(img.get_fdata(), img.affine), anat_dst)
                print(f"    {anat_name} -> anatomicals/raw.nii.gz")
                anat_imported = True
                break
            except Exception as e:
                print(f"    {anat_name}: ERROR - {e}")

    if not anat_imported:
        # Use functional data shape for anatomical placeholder
        func_path = '/mnt/HDD/dataset/NSD_raw/nsddata/ppdata/subj01/func1pt8mm/roi/streams.nii.gz'
        if os.path.exists(func_path):
            img = nib.load(func_path)
            anat_dst = os.path.join(subj_dir, 'anatomicals', 'raw.nii.gz')
            # Create placeholder anatomical from functional shape
            anat_data = np.zeros(img.shape[:3], dtype=np.float32)
            nib.save(nib.Nifti1Image(anat_data, img.affine), anat_dst)
            print(f"    Created placeholder anatomical from functional shape")

    # Create functional transform (with correct reference shape)
    print("\n  Creating transforms...")
    transforms_dir = os.path.join(subj_dir, 'transforms')
    func_path = '/mnt/HDD/dataset/NSD_raw/nsddata/ppdata/subj01/func1pt8mm/roi/streams.nii.gz'

    if os.path.exists(func_path):
        func_img = nib.load(func_path)

        # Create 'func' transform directory (not 'identity' to avoid pycortex override)
        func_dir = os.path.join(transforms_dir, 'func')
        os.makedirs(func_dir, exist_ok=True)

        # Save reference volume (functional data shape)
        ref_file = os.path.join(func_dir, 'reference.nii.gz')
        # Create empty reference with same shape and affine as functional
        ref_data = np.zeros(func_img.shape[:3], dtype=np.float32)
        nib.save(nib.Nifti1Image(ref_data, func_img.affine), ref_file)
        print(f"    reference.nii.gz: shape={func_img.shape[:3]}")

        # Create transform matrix file as JSON (pycortex format)
        xfm_matrix = np.eye(4)
        xfm_file = os.path.join(func_dir, 'matrices.xfm')
        xfm_dict = {
            'coord': xfm_matrix.tolist(),
            'magnet': xfm_matrix.tolist()
        }
        with open(xfm_file, 'w') as f:
            json.dump(xfm_dict, f, indent=2)

        print(f"    'func' transform created")

    # Verify
    print("\n  Verifying installation...")
    try:
        db = cortex.database.db
        # Force reload
        db._subjects = None

        if subject_name in db.subjects:
            print(f"    Subject '{subject_name}' found in database!")

            # Try to load surfaces
            for surf_type in ['wm', 'pia', 'inflated', 'fiducial', 'flat']:
                try:
                    pts, polys = db.get_surf(subject_name, surf_type)
                    print(f"    Surface '{surf_type}': {len(pts)} vertices, {len(polys)} faces")
                except Exception as e:
                    print(f"    Surface '{surf_type}': Not available - {e}")
        else:
            print(f"    WARNING: Subject not in database, may need to restart Python")
    except Exception as e:
        print(f"    Verification error: {e}")

    print("\n" + "="*60)
    print("SETUP COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("  1. Restart Python/IPython kernel")
    print("  2. Run: python scripts/visualize_what_where_3d.py --port 8050")


def main():
    freesurfer_path = '/mnt/HDD/dataset/NSD_raw/nsddata/freesurfer/subj01'

    if not os.path.exists(freesurfer_path):
        print("ERROR: NSD FreeSurfer data not found!")
        print(f"Expected path: {freesurfer_path}")
        return

    setup_pycortex_manual(
        freesurfer_path=freesurfer_path,
        subject_name='subj01'
    )


if __name__ == '__main__':
    main()
