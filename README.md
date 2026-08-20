# DLBS SC-FC Geometry

Reproducible analysis code for the study:

**Cortical geometry dominates structure-function coupling in resting-state fMRI: a multi-model held-out comparison on the Dallas Lifespan Brain Study**

This repository contains the script-based analysis pipeline used to construct structural and functional connectivity data from the Dallas Lifespan Brain Study (DLBS), validate coupling-potential predictors, and perform held-out comparisons of structural, geometric, and hybrid models.

## Overview

The analysis has two main stages:

1. Build subject-level structural connectivity and resting-state functional connectivity matrices from DTI and fMRI data, together with coupling-potential metrics.
2. Run a held-out structure-function comparison using structural network predictors, structural spectral modes, cortical geometric modes, and their hybrid model.

The final comparison includes:

- CP baseline predictors
- Structural connectivity spectral predictors
- Geometric spectral predictors derived from Schaefer parcel centroids
- A hybrid model combining CP, structural, and geometric features
- Component ablations
- K sensitivity analysis
- Hemispheric spin nulls
- Parcel permutation nulls
- Distant-pair sensitivity
- Structural consensus and weighting sensitivity
- Repeated random split robustness
- Optional train/test age balance checks

The workflow uses a fixed random seed by default and keeps the test cohort separated from model fitting.

## Repository layout

```text
dlbs-scfc-geometry/
├── scripts/
│   ├── run_coupling_potential.py   # Stage 1, v2 (affine atlas resampling)
│   ├── dlbs_pipeline_v3.py         # Stage 1, v3 (recommended; ANTs SyN registration)
│   ├── diagnose_qc.py              # QC: separate bad-warp subjects from genuine coverage loss
│   ├── check_reproducibility.py    # QC: process one subject twice and compare
│   └── run_scfc_models.py          # Stage 2
├── results/
│   ├── figures/
│   ├── tables/
│   └── data/
├── data/
├── tests/
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

The `data/` and `results/` directories are placeholders. Raw DLBS data and generated matrices are not included in this repository.

## Requirements

Python 3.10 or newer is recommended.

Main dependencies:

- NumPy
- pandas
- SciPy
- scikit-learn
- matplotlib
- Nilearn
- NiBabel
- DIPY
- ANTsPy (`antspyx`), required by `dlbs_pipeline_v3.py`
- tqdm
- CuPy, optional for GPU acceleration

Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

For GPU runs, install the CuPy package matching the local CUDA installation.

## Input data

The processing script expects BIDS-like DLBS DTI and resting-state fMRI directories.

Set the locations with environment variables:

```bash
export DLBS_FMRI_DIR=/path/to/dlbs_rsfmri/ds004856
export DLBS_DTI_DIR=/path/to/dlbs_dwi/ds004856
```

The processing stage discovers subjects with complete DTI and fMRI data, fetches the 100 parcel Schaefer atlas through Nilearn, and writes subject-level matrices to the output directory.

Precomputed structural and functional connectomes produced by this pipeline are published as a Kaggle dataset: [DLBS Connectome](https://www.kaggle.com/datasets/saeedrezaeiafshar/dlbs-connectome).

## Stage 1: coupling potential validation

Run:

```bash
python scripts/run_coupling_potential.py
```

Useful environment variables:

```bash
export DLBS_OUTPUT_DIR=./results
export DLBS_CHECKPOINT_DIR=./checkpoints
export DLBS_N_SUBJECTS=193
```

The pipeline supports checkpointing and saves structural connectivity, PLV, correlation-based FC, coupling-potential metrics, validation results, and publication figures.

For a smaller test run:

```bash
DLBS_N_SUBJECTS=10 python scripts/run_coupling_potential.py
```

The DTI pipeline uses deterministic tractography, a fractional anisotropy stopping criterion, streamline length filtering, and atlas-based endpoint assignment. fMRI processing applies smoothing and bandpass filtering before parcel time-series extraction and PLV calculation.

## Stage 1, v3: dlbs_pipeline_v3.py (recommended)

`scripts/dlbs_pipeline_v3.py` is a rewrite of Stage 1 that replaces the affine
atlas resampling used in `run_coupling_potential.py` with proper ANTsPy
(`antspyx`) nonlinear (SyN) registration through a T1 anatomical image, and
fixes a transform-inversion bug in the v2 pipeline that misplaced the atlas in
native DWI space. The module docstring documents every correctness, speed,
and quality change versus v2 in detail. `run_coupling_potential.py` is kept
in the repository for reference as the original v2 pipeline.

Because it registers through a T1 image, this pipeline additionally requires
a BIDS-like anatomical directory:

```bash
export DLBS_FMRI_BASE=/path/to/dlbs_rsfmri/ds004856
export DLBS_DTI_BASE=/path/to/dlbs_dwi/ds004856
export DLBS_ANAT_BASE=/path/to/dlbs_anat/ds004856
export DLBS_ATLAS_DIR=/path/to/schaefer_2018   # must contain the Schaefer-100 nii + label files
export DLBS_OUTPUT_DIR=./results
export DLBS_CHECKPOINT_DIR=./checkpoints
```

Subjects are processed in batches for parallel/resumable runs:

```python
import dlbs_pipeline_v3 as P
P.run_batch_pipeline(batch_id=0, n_batches=12)   # full SC + FC pipeline
P.run_sc_only_pipeline(batch_id=0, n_batches=12) # structural connectivity only
P.debug_one_subject("sub-1003")                  # single-subject debug run
```

Set `DLBS_DETERMINISTIC=1` to force single-threaded ANTs registration for
bit-identical reruns (slower). `DLBS_MNI_TEMPLATE` overrides the MNI template
path; otherwise it is fetched through Nilearn.

Structural connectivity is saved with four edge weightings (`count`,
`invlen`, `density`, `mean_length`) instead of raw streamline count alone,
and per-subject QC (atlas warp ratio, framewise displacement, streamline
counts, and more) is written to `<output>/qc/*.json`.

### QC utilities

`scripts/diagnose_qc.py` separates subjects whose registration failed from
parcels with genuine coverage loss, iterating until both sets stabilize, and
writes an exclusion list:

```python
import diagnose_qc as D
result = D.run()                                       # iterate to convergence
result = D.run(label_path="/path/Schaefer...order.txt") # use real parcel names
```

It expects `<DLBS_CONNECTOME_DIR>/group/subjects.txt` and `.../group/sc_count.npy`
(a stacked subjects x parcels x parcels array) plus `.../qc/qc_table.csv`,
which are produced by aggregating the per-subject QC and SC outputs of Stage 1.

`scripts/check_reproducibility.py` reprocesses one subject twice with
`dlbs_pipeline_v3.py` and reports edge-presence agreement, edge-weight
correlation, and coupling-potential metric stability across streamline
thresholds. Run this before launching a full-cohort batch:

```python
import check_reproducibility as R
R.check("sub-1093")
```

## Stage 2: held-out model comparison

The second stage expects the subject-level outputs from Stage 1.

Run:

```bash
python scripts/run_scfc_models.py --data-dir ./results --output-dir ./results/scfc_models
```

The default analysis uses:

- 70/30 train/test split
- Random seed 42
- 60 percent structural edge prevalence for the consensus graph
- 5-fold inner cross-validation
- Ridge regression
- Structural and geometric spectral features
- 100 null iterations for each geometry null

The number of null iterations and repeated splits can be changed for development runs:

```bash
python scripts/run_scfc_models.py \
    --data-dir ./results \
    --output-dir ./results/scfc_models \
    --n-spins 20 \
    --cv-repeats 3
```

If a `participants.tsv` file is available, provide it with:

```bash
python scripts/run_scfc_models.py \
    --data-dir ./results \
    --participants /path/to/participants.tsv
```

## Outputs

The model comparison writes CSV tables containing model performance, ablation results, null distributions, sensitivity analyses, and cross-split robustness results.

Publication figures are written as PNG and PDF files.

A final JSON summary and compressed analysis artifacts are also written to the model output directory.

## Reproducibility

The repository intentionally keeps raw neuroimaging data outside version control. Reproduction therefore requires access to the DLBS data and the corresponding Schaefer atlas resources.

The analysis uses explicit random seeds for cohort splitting, robustness checks, and geometry nulls. Model selection is performed on the training cohort before final evaluation on the held-out cohort.

## Notes

The structural connectivity stage uses affine atlas resampling to the DTI image space. This follows the analysis code used for the study and should not be interpreted as a substitute for nonlinear anatomical registration.

The GPU implementation accelerates parts of PLV and coupling-potential computation when CuPy is available. CPU execution remains supported.

## Author

Saeed Rezaei Afshar
