#!/usr/bin/env python3
"""
Select the “best” CT volume from every subject-level sub-folder and copy it
to the root output directory, renamed with the subject ID.

Criteria
--------
1. Keep only 3-D NIfTI images (ignore 2-D or 4-D).
2. Choose the 3-D image that has the smallest slice thickness
   (hdr.pixdim[3]).
3. If several images tie on slice thickness, prefer the one whose JSON
   metadata contains “brain” and **not** “bone” (case-insensitive).
4. If the best image cannot be unambiguously determined, emit a warning and
   skip that folder.

A matching JSON (same base-name, “.json” extension) is copied alongside the
NIfTI image when present.

Usage
-----
Edit ROOT to point at your top-level directory (e.g. /Volumes/TB5/CT/output/)
and run:

    python pick_best_ct.py
"""

import json
import logging
import os
import shutil
from pathlib import Path
import argparse
import sys
from deshear import deshear_nifti
import nibabel as nib

LOG_LEVEL = logging.INFO                   # INFO or DEBUG for more detail
OUT_EXT = ".nii"                           # output images get this extension

logging.basicConfig(
    format="%(levelname)s: %(message)s",
    level=LOG_LEVEL,
)

def is_nifti(path: Path) -> bool:
    return path.suffix in {".nii", ".nii.gz"}

def load_header(path: Path):
    """Read NIfTI header only (no large data array)."""
    return nib.load(str(path), mmap=False).header

def series_description(json_file: Path) -> str:
    if json_file.exists():
        try:
            with open(json_file, "r") as f:
                meta = json.load(f)
            return str(meta.get("SeriesDescription", "")).lower()
        except Exception:
            pass
    return ""

def has_all_keywords(text: str, keywords: list[str]) -> bool:
    text_upper = text.upper()
    return all(k.upper() in text_upper for k in keywords)


def select_best_image(folder: Path) -> Path | None:
    """Return Path to the chosen image or None if no clear winner."""
    candidates = []

    for img in filter(is_nifti, folder.iterdir()):
        try:
            hdr = load_header(img)
        except Exception as e:
            logging.warning("Cannot read %s (%s); skipping.", img.name, e)
            continue
        if hdr["dim"][0] != 3:
            continue
        if hdr["dim"][3] < 2:
            continue
        slice_thickness = float(hdr["pixdim"][3])
        if slice_thickness > 9:
          continue
        total_height = hdr["dim"][3] * slice_thickness
        if total_height < 105:
            continue
        json_path = img.with_suffix(".json")
        if not json_path.exists():
            continue
        desc = series_description(json_path)
        desc_upper = desc.upper()
        # exclude Philips neck images
        if has_all_keywords(desc_upper, ["CTA", "NECK"]):
            continue  # exclude this image
        if has_all_keywords(desc_upper, ["BONE"]):
            continue  # exclude this image
        kernel = ""
        EXCLUDE_KERNELS = {"YB", "D"}
        series_num = -1
        if json_path.exists():
            try:
                with open(json_path) as f:
                    meta = json.load(f)
                # exclude GE neck images
                protocol = str(meta.get("ProtocolName", ""))
                # if has_all_keywords(protocol, ["CAROTID"]):
                #    continue
                if has_all_keywords(protocol, ["CTA", "NECK"]):
                    continue
                if has_all_keywords(protocol, ["CT", "ANGIOGRAM", "NECK"]):
                    continue
                # Philips neck images
                body_part = str(meta.get("BodyPartExamined", ""))
                if "CAROTID" in body_part.upper():
                    continue
                kernel = str(meta.get("ConvolutionKernel", "")).upper()
                if kernel in EXCLUDE_KERNELS:
                  continue
                series_num = int(meta.get("SeriesNumber", -1))
            except Exception:
                pass

        candidates.append(
            {
                "path": img,
                "thk": slice_thickness,
                "desc": desc,
                "is_brain": ("bone" not in desc),
                "is_thins": "thins" in desc,
                "kernel": kernel,
                "is_axial": "AX" in desc_upper,
                "series_num": series_num,
            }
        )

    if not candidates:
        return None
    # Exclude kernel-less candidates if others have kernels
    has_kernel = [c for c in candidates if c["kernel"]]
    if has_kernel:
        candidates = has_kernel
    
    min_thk = min(c["thk"] for c in candidates)
    thinnest = [c for c in candidates if c["thk"] == min_thk]

    filtered = [c for c in thinnest if c["is_brain"]]
    if not filtered:
        filtered = thinnest

    thins = [c for c in filtered if c["is_thins"]]
    chosen_set = thins or filtered

    if len(chosen_set) == 1:
        return chosen_set[0]["path"]

    ub_kernel = [c for c in chosen_set if c["kernel"] == "UB"]
    if ub_kernel:
      return ub_kernel[0]["path"]

    ub_kernel = [c for c in chosen_set if c["kernel"] == "D"]
    if len(ub_kernel) == 1:
        return ub_kernel[0]["path"]


    axial = [c for c in chosen_set if c["is_axial"]]
    if len(axial) == 1:
        return axial[0]["path"]

    # Final tiebreaker: highest SeriesNumber
    best_series = max(c["series_num"] for c in chosen_set)
    best_series_set = [c for c in chosen_set if c["series_num"] == best_series]
    if len(best_series_set) == 1:
        return best_series_set[0]["path"]

    logging.warning(
        "Ambiguous choice in %s; skipping. Candidates:\n%s",
        folder.name,
        "\n".join(
            f"  {c['path'].name} (thk={c['thk']}, desc='{c['desc']}', kernel='{c['kernel']}', series={c['series_num']})"
            for c in chosen_set
        ),
    )
    return None

def find_preferred_variant(base_img: Path) -> Path:
    """
    Return the preferred variant of a NIfTI file, if available.
    Checks for *_Tilt*, *_Eq*, or *_Tilt_Eq* variants.
    """
    base_stem = base_img.stem
    suffix = ''.join(base_img.suffixes)
    parent = base_img.parent

    # Search for files like "base_Tilt_Eq_1.nii.gz" or "base_Tilt_1.nii.gz"
    candidates = sorted(parent.glob(f"{base_stem}_*{suffix}"))
    for c in candidates:
        name = c.stem
        if "_Tilt" in name and "_Eq" in name:
            return c
    for c in candidates:
        if "_Tilt" in c.stem:
            return c
    for c in candidates:
        if "_Eq" in c.stem:
            return c
    return base_img

def main(input_dir: Path, output_dir: Path):
    if not input_dir.exists():
        logging.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    for child in input_dir.iterdir():
        if not child.is_dir():
            continue
        if child.name.startswith("_"):
            continue
        best_img = select_best_image(child)
        if not best_img:
            continue
        
        # Save base stem for JSON lookup before selecting variant
        base_stem = best_img.stem
        variant_img = find_preferred_variant(best_img)
        if variant_img != best_img:
            logging.info(f"Using variant for {child.name}: {variant_img.name}")
        best_img = variant_img

        dest_img = output_dir / f"{child.name}{OUT_EXT}"
        shutil.copy2(best_img, dest_img)
        deshear_nifti(str(dest_img))
        # Always get JSON using the base stem
        src_json = best_img.parent / f"{base_stem}.json"
        if src_json.exists():
            dest_json = dest_img.with_suffix(".json")
            shutil.copy2(src_json, dest_json)
        logging.info("Saved %s", dest_img.relative_to(output_dir))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select the best CT volume from each subject folder."
    )
    parser.add_argument(
        "input_dir", nargs="?", default="./raw", help="Input folder with subdirectories of NIfTI images"
    )
    parser.add_argument(
        "output_dir", nargs="?", default="./best", help="Output folder for best images"
    )
    parser.add_argument(
        "--make-jpgs", dest="make_jpgs", action="store_true", default=True,
        help="Generate JPEGs from selected images (default: True)"
    )
    parser.add_argument(
        "--no-make-jpgs", dest="make_jpgs", action="store_false",
        help="Do not generate JPEGs"
    )
    args = parser.parse_args()

    main(Path(args.input_dir), Path(args.output_dir))

    if args.make_jpgs:
        try:
            import subprocess
            subprocess.run(
                [sys.executable, "dir2jpg.py", str(Path(args.output_dir))],
                check=True
            )
        except Exception as e:
            logging.error("Failed to run dir2jpg.py: %s", e)
