#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
from SYNcro import normalize

def bids_ct_files(bids_dir):
    """Yield tuples of (subject_id, CT image path) from BIDS directory."""
    for sub_dir in sorted(Path(bids_dir).glob("sub-*")):
        ct_dir = sub_dir / "dwi"
        if not ct_dir.is_dir():
            continue
        for ct_img in ct_dir.glob("*_dwi.nii*"):
            yield sub_dir.name.replace("sub-", ""), ct_img

def main(bids_dir: Path):
    if not bids_dir.is_dir():
        print(f"ERROR: BIDS directory does not exist: {bids_dir}")
        sys.exit(1)

    out_dir = bids_dir / "derivatives" / "syncro"
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = list(bids_ct_files(bids_dir))
    if not subjects:
        print("No DWI images found in BIDS structure.")
        return

    print(f"Found {len(subjects)} CT scans. Processing...")

    for subj, ct_path in subjects:
        base = ct_path.stem.replace(".nii", "")  # remove possible double extension
        warped_ct = out_dir / f"w{ct_path.name}"
        warped_bt1 = out_dir / f"wbt1{ct_path.name}"

        bids_ct = out_dir / f"sub-{subj}_space-MNI152NLin6Asym_dwi.nii.gz"
        bids_bt1 = out_dir / f"sub-{subj}_desc-synth_brain_space-MNI152NLin6Asym_T1w.nii.gz"

        if bids_ct.exists() and bids_bt1.exists():
            print(f"Skipping sub-{subj} (already normalized)")
            continue

        print(f"Normalizing {ct_path.name} for sub-{subj}")
        try:
            normalize([str(ct_path)], str(out_dir), is_ct=True)

            # Rename outputs to BIDS format
            if warped_ct.exists():
                warped_ct.rename(bids_ct)
            else:
                print(f"Warning: warped CT not found for sub-{subj}")

            if warped_bt1.exists():
                warped_bt1.rename(bids_bt1)
            else:
                print(f"Warning: warped T1 not found for sub-{subj}")

        except Exception as e:
            print(f"Failed to normalize {ct_path.name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize BIDS CT scans using SYNcro.")
    parser.add_argument("bids_dir", nargs="?", default="./bids", help="Path to BIDS root folder")
    args = parser.parse_args()
    main(Path(args.bids_dir))
