#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path
import json
import sys

def make_bids_boilerplate(bids_dir: Path):
    bids_dir.mkdir(parents=True, exist_ok=True)

    # dataset_description.json
    desc = bids_dir / "dataset_description.json"
    if not desc.exists():
        desc.write_text(json.dumps({
            "Name": "DWI Stroke Imaging Dataset",
            "BIDSVersion": "1.8.0",
            "DatasetType": "raw",
            "Authors": ["Your Name"]
        }, indent=2))
        print(f"Created {desc}")

    # participants.json
    participants_json = bids_dir / "participants.json"
    if not participants_json.exists():
        participants_json.write_text(json.dumps({
            "participant_id": {
                "Description": "Unique participant ID"
            }
        }, indent=2))
        print(f"Created {participants_json}")

    # participants.tsv
    participants_tsv = bids_dir / "participants.tsv"
    if not participants_tsv.exists():
        participants_tsv.write_text("participant_id\n")
        print(f"Created {participants_tsv}")

def copy_anon_to_bids(input_dir: Path, output_dir: Path):
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    modality = "dwi"
    nii_files = sorted(input_dir.glob(f"*_{modality}.nii*"))
    for nii in nii_files:
        subj = nii.stem.split("_")[0].replace(f"{modality}", "").strip("_")
        if not subj.isdigit():
            print(f"Skipping unrecognized subject ID in {nii.name}")
            continue
        bids_subj = f"sub-{int(subj)}"
        bids_ct_dir = output_dir / bids_subj / "dwi"
        bids_ct_dir.mkdir(parents=True, exist_ok=True)

        # Copy .nii.gz
        ext = ".nii.gz" if nii.name.endswith(".nii.gz") else ".nii"
        bids_img = bids_ct_dir / f"{bids_subj}_{modality}{ext}"
        shutil.copy2(nii, bids_img)

        # Copy .json
        stem = nii.name.replace(".nii.gz", "").replace(".nii", "")
        json_src = input_dir / f"{stem}.json"
        if json_src.exists():
            bids_json = bids_ct_dir / f"{bids_subj}_{modality}.json"
            shutil.copy2(json_src, bids_json)

        print(f"Copied {nii.name} -> {bids_img.name}")

def main(input_dir: Path, output_dir: Path):
    if not input_dir.is_dir():
        print(f"ERROR: Input directory does not exist: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    make_bids_boilerplate(output_dir)
    copy_anon_to_bids(input_dir, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert anonymized DWI images into BIDS structure.")
    parser.add_argument("input_dir", nargs="?", default="./anon", help="Folder containing *_ct.nii(.gz) and JSON")
    parser.add_argument("output_dir", nargs="?", default="./bids", help="Target BIDS folder")
    args = parser.parse_args()

    main(Path(args.input_dir), Path(args.output_dir))
