#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import sys
import shutil
import subprocess

def check_dcm2niix_available():
    if shutil.which("dcm2niix") is None:
        print("Error: 'dcm2niix' is not found in your system PATH.")
        print("Please install it: https://github.com/rordenlab/dcm2niix")
        sys.exit(1)

def main(input_dir, output_dir="./raw"):
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    # Final warning if output_dir already existed
    output_preexists = os.path.exists(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "dcm2niix",
        "-f", "%g-%i/%s_%p",        # Output filename pattern
        "-w", "1",
        "-m", "y",

        "-o", output_dir,
        input_dir
    ]

    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("dcm2niix failed with error:", e)
        sys.exit(1)
    if output_preexists:
        print("\n" + "*" * 70)
        print(f"WARNING: Output directory already existed: {output_dir}")
        print("   Existing files were NOT overwritten by dcm2niix.")
        print("*" * 70 + "\n")

if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser(
        description="Select the best CT volume from each subject folder."
    )
    parser.add_argument(
        "input_dir", nargs="?", default="./DICOM", help="Input folder with subdirectories of DICOM images"
    )
    parser.add_argument(
        "output_dir", nargs="?", default="./raw", help="Output folder for NIfTI images"
    )
    args = parser.parse_args()

    main(Path(args.input_dir), Path(args.output_dir))