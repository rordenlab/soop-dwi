#!/usr/bin/env python3

from pathlib import Path
import argparse
import brainchop  # Ensure installed
import csv
import json
import nibabel as nib
import numpy as np
import random
import shutil
import subprocess
import sys


def compute_nonzero_bounds(data, is_ct):
    if is_ct:
        mask = data > -999
    else:
        mask = data > 0
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None, mask
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    slices = tuple(slice(start, end) for start, end in zip(mins, maxs))
    return slices, mask

def remove_zero_margins(img_path, out_path="", is_ct=False, save_mask=False, recenter_origin=False):
    img = nib.load(str(img_path))
    data = img.dataobj[:] if is_ct else img.get_fdata()
    original_shape = data.shape
    slices, mask = compute_nonzero_bounds(data, is_ct)
    if save_mask:
        mask_img = nib.Nifti1Image(mask.astype(np.uint8), img.affine, header=img.header)
        mask_path = img_path.parent / f"mask_{img_path.name}"
        nib.save(mask_img, str(mask_path))
        print(f"Saved mask to {mask_path}")
    if slices is None:
        print(f"{img_path.name}: all voxels excluded — skipping.")
        return
    cropped_data = data[slices]
    cropped_shape = cropped_data.shape
    if cropped_shape == original_shape:
        print(f"{img_path.name}: no zero/air borders detected — skipping.")
        return
    crop_offset_voxel = [s.start for s in slices]
    new_affine = img.affine.copy()
    new_affine[:3, 3] = nib.affines.apply_affine(img.affine, crop_offset_voxel)
    if recenter_origin:
        center_voxel = (np.array(cropped_data.shape) - 1) / 2.0
        rotation = new_affine[:3, :3]
        translation = -rotation @ center_voxel
        new_affine[:3, 3] = translation
    cropped_img = nib.Nifti1Image(cropped_data, new_affine, header=img.header)
    cropped_img.set_data_dtype(img.get_data_dtype())
    if not out_path:
        out_path = img_path.parent / f"z{img_path.name}"
    nib.save(cropped_img, str(out_path))

    def shape_str(shape):
        return '×'.join(str(d) for d in shape)

    print(f"Cropped {shape_str(original_shape)} -> {shape_str(cropped_shape)} and saved as {out_path.name} recenter: {recenter_origin}")

def do_brainchop(input_path, out_path, is_ct=False, border=0, model="mindgrab"):
    cmd = ["brainchop", "-m", model, "-i", input_path, "-o", out_path]
    if is_ct:
        cmd.append("--ct")
    if border > 0:
        cmd += ["-b", str(border)]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"brainchop failed: {e}")
    # if overwrite requested, handle brainchop conversion .nii -> .nii.gz
    if input_path == out_path:
        input_path = Path(input_path)
        print(input_path)
        if input_path.suffix == ".nii" and input_path.with_suffix(".nii.gz").exists():
            if input_path.exists():
                input_path.unlink()  # delete the .nii file

def main(src_dir: Path, dst_dir: Path):
    modality = "dwi"
    border = 25
    random.seed(42)
    if not src_dir.exists():
        print(f"ERROR: Input directory does not exist: {src_dir}")
        sys.exit(1)
    dst_dir.mkdir(parents=True, exist_ok=True)
    lookup_file = dst_dir / "lookup.tsv"
    nii_files = sorted([f for f in src_dir.glob("*.nii*") if f.with_suffix(".json").exists()])

    
    random.shuffle(nii_files)
    pad_width = len(str(len(nii_files)))

    with open(lookup_file, "w", newline="") as tsvfile:
        writer = csv.writer(tsvfile, delimiter="\t")
        writer.writerow(["original_name", "anonymized_name"])

        for i, nii in enumerate(nii_files, start=1):
            anon_id = f"{i:0{pad_width}}_{modality}"
            new_nii = dst_dir / f"{anon_id}{nii.suffix}"
            new_json = dst_dir / f"{anon_id}.json"
            old_json = nii.with_suffix(".json")

            shutil.copy2(nii, new_nii)
            do_brainchop(new_nii, new_nii, modality == "ct", border)
            nii_path = new_nii
            if not nii_path.exists() and nii_path.suffix == ".nii":
                gz_path = nii_path.with_suffix(".nii.gz")
                if gz_path.exists():
                    nii_path = gz_path
            remove_zero_margins(nii_path, nii_path, modality == "ct", False, True)
            shutil.copy2(old_json, new_json)

            try:
                with open(new_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "AcquisitionTime" in data:
                    del data["AcquisitionTime"]
                    with open(new_json, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
            except Exception as e:
                print(f"Warning: Could not sanitize {new_json}: {e}")

            writer.writerow([nii.stem, anon_id])

    print(f"Copied {len(nii_files)} image pairs to {dst_dir}")
    print(f"Audit log saved to {lookup_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anonymize filenames and clean metadata.")
    parser.add_argument("input_dir", nargs="?", default="./best", help="Input folder with NIfTI + JSON pairs")
    parser.add_argument("output_dir", nargs="?", default="./anon", help="Output folder for anonymized files")
    args = parser.parse_args()

    main(Path(args.input_dir), Path(args.output_dir))
