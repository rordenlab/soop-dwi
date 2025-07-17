#!/usr/bin/env python3
import sys
import nibabel as nib
import numpy as np

ANGLE_TOLERANCE_DEGREES = 0.0001

def deshear_affine(affine):
    """
    Remove shear by converting to a quaternion-based qform and back.
    """
    hdr = nib.Nifti1Header()
    hdr.set_qform(affine, code=1)
    return hdr.get_qform()

def angle_between(v1, v2):
    dot = np.dot(v1, v2)
    return np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))

def max_shear_deviation(affine):
    """
    Returns the maximum deviation from 90° between any two axes.
    """
    A = affine[:3, :3]
    norms = np.linalg.norm(A, axis=0)
    unit_axes = A / norms
    max_dev = 0.0
    for i in range(3):
        for j in range(i+1, 3):
            angle = angle_between(unit_axes[:, i], unit_axes[:, j])
            deviation = abs(angle - 90.0)
            max_dev = max(max_dev, deviation)
    return max_dev

def deshear_nifti(filename):
    img = nib.load(filename)
    orig_affine = img.affine

    deviation = max_shear_deviation(orig_affine)
    if deviation <= ANGLE_TOLERANCE_DEGREES:
        print(f"No significant shear detected in {filename}. Max deviation: {deviation:.6f}°")
        return

    print(f"Shear detected in {filename} (max deviation {deviation:.6f}°), correcting...")

    new_affine = deshear_affine(orig_affine)
    data = np.asanyarray(img.dataobj)
    new_img = nib.Nifti1Image(data, new_affine, header=img.header.copy())
    new_img.set_sform(new_affine, code=1)
    new_img.set_qform(new_affine, code=1)
    new_img.set_data_dtype(img.get_data_dtype())
    new_img.update_header()
    nib.save(new_img, filename)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python deshear.py myImage.nii")
        sys.exit(1)
    deshear_nifti(sys.argv[1])
