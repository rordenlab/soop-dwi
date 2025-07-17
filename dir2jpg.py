#!/usr/bin/env python3

import sys
from pathlib import Path
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def robust_normalize(slice_data, lower_pct=2, upper_pct=98):
    """Normalize based on robust percentiles (like FSL's robust range)."""
    vmin, vmax = np.percentile(slice_data, [lower_pct, upper_pct])
    slice_data = np.clip(slice_data, vmin, vmax)
    norm = (slice_data - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(slice_data)
    return (norm * 255).astype(np.uint8)

def save_middle_slice_as_jpeg(nifti_path, output_dir):
    img = nib.load(str(nifti_path))
    data = img.get_fdata()

    # Get middle axial slice
    z = data.shape[2] // 2
    slice_data = data[:, :, z]

    # Normalize using robust range
    slice_img = robust_normalize(slice_data)

    # Save as JPEG
    name = nifti_path.name
    if name.endswith(".nii.gz"):
        base = name[:-7]  # remove '.nii.gz'
    elif name.endswith(".nii"):
        base = name[:-4]  # remove '.nii'
    else:
        base = nifti_path.stem
    output_path = output_dir / f"{base}.jpg"
    plt.imsave(str(output_path), slice_img, cmap='gray', format='jpeg')

def main():
    if len(sys.argv) != 2:
        print("Usage: python dir2jpg.py /path/to/niftis")
        sys.exit(1)

    folder = Path(sys.argv[1])
    if not folder.is_dir():
        print(f"Error: {folder} is not a valid directory.")
        sys.exit(1)

    nifti_files = sorted(
        f for f in folder.iterdir()
        if f.suffix in [".nii", ".gz"] and (f.name.endswith(".nii") or f.name.endswith(".nii.gz"))
    )
    if not nifti_files:
        print("No NIfTI files found in the folder.")
        return

    for f in nifti_files:
        try:
            save_middle_slice_as_jpeg(f, folder)
            print(f"Saved JPEG for {f.name}")
        except Exception as e:
            print(f"Failed to process {f.name}: {e}")

if __name__ == "__main__":
    main()
