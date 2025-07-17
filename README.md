## About

This repository provides tools for converting DWI scans from DICOM format to NIfTI using BIDS organization. The goal is to support scientific and medical discovery by enabling standardized analysis pipelines. While the DICOM format is the clinical standard for medical imaging, it is complex and not broadly supported by most neuroimaging tools. In contrast, the NIfTI format is simpler, and the Brain Imaging Data Structure (BIDS) allows automated and reproducible analysis.

## Caveats

This script uses emerging methods, so users should be aware of a few caveats:

 - Open source DICOM datasets of stroke are scarce. Here a set is provided with TRACE scans that show the lesion well. It is preferrable to have both the b=0 and TRACE images from the same series, as the former shows good tissue contrast and the latter shows the injury. This limitation reduces the subsequent quality of the conversion.
 - These scripts require Python 3.10 or later (due to the BrainChop dependency).
 - The `bids2norm` script depends on FreeSurfer's SynthSR module. This requires installing the full FreeSurfer software stack and license. Alternatively, SynthSR can be installed separately, but has [complex dependencies](https://github.com/BBillot/SynthSR).

## Installation

Install the scripts and example data by running the following:

```
git clone https://github.com/rordenlab/soop-dwi.git
cd soop-dwi
pip install -r requirements.txt
unzip DICOM.zip
```

If you wish to use `bids2norm`, you must also install [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall) and its license file.

## Usage

The repository includes a sample DICOM dataset for testing. The following commands process this example dataset:

```
python 1dcm2raw.py ./DICOM ./raw
python 2raw2best.py ./raw ./best
# Check generated JPGs to ensure correct image selection
python 3best2anon.py ./best ./anon
python 4anon2bids.py ./anon ./bids
# Optional group-level analysis
python 5bids2norm.py ./bids
python nii2mean.py ./bids/derivatives/syncro _dwi.nii.gz
```

It is advisable to inspect the results after each stage. This modular pipeline allows users to intervene or adjust steps based on the complexities of real-world clinical data.

### Processing Stages

The diagram below shows the six processing stages. Arrows indicate transformation from input to output.

**Figure 1** A: Convert DICOM to NIfTI. B: Select best DWI series. C: Anonymize and crop. D: Convert to BIDS. E: Spatial normalization.


1. **`1dcm2raw.py [input_dir] [output_dir]`**
   Converts all DICOM images to NIfTI format. Files are stored in folders named by accession number. Series are named using the series number and protocol name. Interpolation corrects for variable slice thickness and gantry tilt.

2. **`2raw2best.py [input_dir] [output_dir]`** 
   Selects the best series from each study based on slice thickness, field of view, and soft-tissue convolution kernel. Creates corresponding JPEGs for quality control.
   - Uses helper script `dir2jpg.py` to generate previews.
   - The helper script `deshear.py` removes rounding errors that can be detected as [shear](https://github.com/rordenlab/dcm2niix/issues/945).

3. **`3best2anon.py [input_dir] [output_dir]`**  
   Renames files using anonymized IDs. Generates a `lookup.tsv` file to map accession numbers to anonymized IDs. This file should be kept secure. Bis and brain-extracts each image to remove facial features and de-identify data. By default, the mask is dilated 25mm beyond the brain boundary. This script has three constants that you may want to adjust
   - `modality = "ct"` will define the BIDS modality.
   - `border = 25` specifies dilation of brain extraction. Smaller values remove scalp features.
   - `random.seed(42)` can be changed so identical inputs will yield a different scrambled ID.

4. **`4anon2bids.py [input_dir] [output_dir`]**  
   Converts cropped images to BIDS format. Also generates placeholder boilerplate files:
   - `dataset_description.json`
   - `participants.tsv`
   - `participants.json`

5. **`5bids2norm.py [bids_dir]`**  
   Normalizes each image to a common space (MNI152). Saves outputs in the `derivatives/syncro` directory.
   - Calls `SYNcro.py` to register each image using ANTs.
   - Uses FreeSurfer's SynthSR to synthesize a T1-weighted image from CT.
   - Uses brainchop for an approximate brain extraction.
   - Uses ANTS to align images to standard space
   - Images aligned to the FSL template `MNI152_T1_1mm_brain.nii.gz`

7. **`nii2mean.py [input_dir] [filter]`**  
   Creates a group-average image from all files in the input folder matching the specified filter.



## Links

 - DICOM.zip images: [Acute Ischemic Infarct Segmentation](https://github.com/GriffinLiang/AISD)
 - DICOM-to-NIfTI conversion: [dcm2niix](https://github.com/rordenlab/dcm2niix)
 - Core Python dependencies: [numpy](https://github.com/numpy/numpy), [nibabel](https://github.com/nipy/nibabel)
 - Spatial normalization: [ANTS](https://pubmed.ncbi.nlm.nih.gov/17659998/)
 - Brain extraction: [BrainChop](https://github.com/neuroneural/brainchop-cli)
 - T1 synthesis: [SynthSR](https://surfer.nmr.mgh.harvard.edu/fswiki/SynthSR)

## Related

 - [Acute Ischemic Infarct Segmentation](https://github.com/GriffinLiang/AISD) provides sample segmentations

## Citation

 - Absher, J., ... Rorden, C. (in prep). *SOOP-CT: Acute Stroke CT with Open Tools for De-Identification and Sharing*
