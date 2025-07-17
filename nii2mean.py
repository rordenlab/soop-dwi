#!/usr/bin/env python3

import sys
import os
from pathlib import Path
import nibabel as nib
import numpy as np

def create_mean_image(nii_paths, output_file=None):
    if not nii_paths:
        print("ERROR: No matching NIfTI files found.")
        sys.exit(1)

    n = len(nii_paths)
    if output_file is None:
        output_file = f"./mean_of_{n}.nii.gz"
    else:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    print(f"Found {n} file(s). Calculating mean...")

    ref_img = nib.load(str(nii_paths[0]))
    data_sum = np.zeros_like(ref_img.get_fdata(), dtype=np.float32)

    for path in nii_paths:
        img = nib.load(str(path))
        data_sum += img.get_fdata()

    mean_data = data_sum / n
    mean_img = nib.Nifti1Image(mean_data, ref_img.affine, ref_img.header)
    nib.save(mean_img, output_file)

    print(f"Mean of {n} image(s) saved to: {output_file}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python nii2mean.py <input_dir> <suffix_filter> [output_file]")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    suffix = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else None

    if not input_dir.is_dir():
        print(f"ERROR: Not a directory: {input_dir}")
        sys.exit(1)

    nii_paths = sorted([
        f for f in input_dir.iterdir()
        if f.is_file() and f.name.endswith(suffix) and (f.suffix == '.nii' or f.suffixes[-2:] == ['.nii', '.gz'])
    ])

    create_mean_image(nii_paths, output_file)

if __name__ == "__main__":
    main()
