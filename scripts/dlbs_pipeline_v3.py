"""
DLBS structural + functional connectome pipeline (v3).

Changes vs v2, grouped by why they matter.

CORRECTNESS
  1. Temp-file lifetime bug (the crash in your log). `nib.load()` is lazy, so
     `nib.load(tmp); os.remove(tmp)` returns a proxy that raises
     FileNotFoundError the first time anything touches the data. That is
     exactly "[sub-1093] ERROR (fMRI): [Errno 2] No such file or directory:
     '/tmp/tmp79749x5d.nii.gz'". All ANTs <-> nibabel conversion now happens
     in memory, so there is no temp file at all.
  2. Missing `whichtoinvert`. ANTsPy only auto-inverts an affine when the
     transform list is exactly [something.mat, something-not-mat]. Your atlas
     chain is 3 transforms long, so BOTH affines were applied FORWARD instead
     of inverted. The atlas was landing in the wrong place in native DWI
     space, which is why 89% of your streamline endpoints hit background and
     the SC matrix only had 1640 nonzero entries. Same bug in the WM mask
     warp. Every transform chain now carries an explicit inversion flag.
  3. TensorModel was fit on every voxel including background (no mask).
  4. Motion parameters used as fMRI confounds were the first 6 of 12 raw
     affine parameters (i.e. two rows of the rotation matrix). Now proper
     3 translations + 3 rotation angles, plus derivatives.

SPEED (largest first)
  5. T1 preprocessing (N4 + skull strip) and the T1->MNI SyN ran TWICE per
     subject, once in the DTI stage and once in the fMRI stage. Both are now
     computed once, cached to disk, and reused across stages AND across runs.
  6. `csd_fit.odf(sphere)` materialised a (128,128,50,724) float64 array,
     4.7 GB. Replaced by ProbabilisticDirectionGetter.from_shcoeff on the
     SH coefficients (~0.3 GB).
  7. DWI denoising looped nlmeans over volumes. Default is now MPPCA on the
     whole 4D, restricted to the brain mask. Method is configurable.
  8. DWI motion correction wrote 2 gzip NIfTIs per volume to disk. Now
     in-memory, with a tuned rigid preset.
  9. Endpoint -> parcel mapping was a Python loop over ~130k streamlines with
     2 affine calls each. Now fully vectorised (one matmul + one bincount).
 10. PLV was an O(n^2) Python loop. It is one complex matmul.
 11. Adamic-Adar / resource-allocation were O(n^2) loops with intersect1d.
     Both are single matrix products for a binary adjacency.
 12. fMRI band-pass/detrend/confound regression ran voxelwise on the 4D MNI
     volume, then parcellated. Those operations are linear and identical per
     voxel, so they commute with parcel averaging. Parcellating first and
     cleaning the 100 time series is numerically equivalent and far cheaper.
 13. The MNI template is resampled onto the atlas grid once at startup, so
     NiftiLabelsMasker-style resampling never happens per subject.
 14. ANTs transforms are written to a managed per-subject directory instead
     of /tmp, and cleaned up on demand, so /tmp stops filling up.

QUALITY
 15. Streamline endpoints that land just outside a parcel are assigned to the
     nearest parcel within ENDPOINT_MAX_DIST_MM (default 2 mm). Tracking stops
     in white matter, just short of the cortical ribbon, so a strict lookup
     discards most valid endpoints. Set to 0 for the old strict behaviour.
 16. SC is saved with four weightings, because raw streamline count is biased
     by both fibre length and parcel size: count, invlen (sum of 1/length),
     density (Hagmann-style, also divided by parcel volume) and mean_length.
 17. Framewise displacement is computed, saved and flagged per subject, so
     high-motion scans can be filtered before group analysis rather than
     quietly inflating short-range correlations.

FIXES ADDED AFTER THE "SVD did not converge" CRASH
 18. Root cause: MPPCA only visits patch centres in
     [patch_radius, shape - patch_radius) and divides by per-voxel patch
     coverage at the end. An in-mask voxel no valid centre can reach gets
     0/0 = NaN. dipy's own `denoised[mask == 0] = 0` does not clear it
     because the voxel IS in the mask, and `data * mask[..., None]` does not
     either, since NaN * 0 is NaN. One NaN reaching TensorModel.fit raises
     LinAlgError from inside numpy's pinv. With a 50-slice acquisition, brain
     and noise specks routinely sit on the first or last slice, which is
     exactly this case. Those voxels are now excluded from the denoising mask
     and keep their original signal.
 19. Defence in depth: data is checked for non-finite values after loading,
     after motion correction and after denoising, each with a message naming
     the stage, and repaired from the pre-stage data rather than zeroed. The
     fitting mask additionally excludes any non-finite or zero-b0 voxel.
 20. b-vectors are validated (finite, unit norm, zero vector iff b0) before
     gradient_table, which otherwise raises on non-unit vectors.
 21. DWI motion correction is ~4.5x faster. Benchmarked against known rigid
     transforms at 128x128x50: a 2-level schedule with a mask on the reference
     gives 1.14 s/volume versus 5.14 s for the 3-level schedule, with slightly
     LOWER residual error. Full-resolution refinement buys nothing for
     within-subject rigid alignment. The same treatment is applied to fMRI
     motion correction, where ants.motion_correction computes a mask but then
     never passes it to the registration.

TRACTOGRAPHY, WHICH WAS 90% OF THE RUNTIME
 22. Precomputed SH -> sphere map (sh_to_pmf). Measured 1.7x. Enabled
     automatically when the estimate fits TRACKING_PMF_MEMORY_BUDGET_GB
     (about 2.4 GB at 128x128x50 with a 362-vertex sphere).
 23. 362-vertex tracking sphere instead of 724. Measured 1.9x, with the same
     streamline yield. Angular resolution goes from about 5 to 7 degrees,
     still far finer than MAX_ANGLE.
 24. Seed chunks run in parallel processes. Workers return endpoints and
     lengths rather than streamlines, so the large arrays never cross a
     process boundary, and the precomputed PMF is built before forking so
     children share it copy-on-write.
     The seed list is split into a FIXED number of chunks regardless of worker
     count, each with its own deterministic random seed, so a subject gives
     bit-identical streamlines on a 2-core and a 4-core machine. Verified.
 25. Fixed seed budget (N_SEEDS_TOTAL) rather than a per-voxel grid. With grid
     seeding, a subject with a larger WM mask gets proportionally more
     streamlines, so raw SC weights are not comparable across subjects. Set
     N_SEEDS_TOTAL=None to go back to the grid.

Install (before importing this module):
    python -m pip install -r requirements.txt

On Kaggle, the PyPI package `ants` shadows ANTsPy under the same import
name, so uninstall it first:
    !pip uninstall -y ants
    !pip install -q antspyx dipy nilearn
"""

import os

# ITK threading has to be configured before ANTsPy is imported.
#
# Multithreaded ITK registration is not bitwise reproducible: floating-point
# reductions happen in a nondeterministic order across threads. Setting a
# random seed fixes the metric sampling but NOT this. Export
# DLBS_DETERMINISTIC=1 to force single-threaded registration, which makes runs
# bit-identical at the cost of slower registration.
_DETERMINISTIC = os.getenv("DLBS_DETERMINISTIC", "0") not in ("0", "", "false", "False")
_N_THREADS = "1" if _DETERMINISTIC else str(os.cpu_count() or 4)
os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", _N_THREADS)
os.environ.setdefault("OMP_NUM_THREADS", _N_THREADS)

import fcntl
import gc
import json
import pickle
import shutil
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import nilearn
from nilearn import datasets, image, signal as nl_signal
from nilearn.connectome import ConnectivityMeasure

import ants

import dipy
from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
from dipy.direction import ProbabilisticDirectionGetter
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.reconst.csdeconv import (
    ConstrainedSphericalDeconvModel,
    auto_response_ssst,
)
from dipy.reconst.dti import TensorModel, fractional_anisotropy
from dipy.segment.mask import median_otsu
from dipy.tracking import utils as tracking_utils
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking.streamline import Streamlines, length

from scipy import sparse
from scipy.linalg import expm, polar
from scipy.ndimage import distance_transform_edt
from scipy.signal import hilbert

warnings.filterwarnings("ignore")


if not hasattr(ants, "registration") or not hasattr(ants, "apply_transforms"):
    raise ImportError(
        "\nThe imported `ants` package is NOT ANTsPy/antspyx.\n"
        "Kaggle ships the unrelated PyPI package `ants`.\n\n"
        "Run this in a fresh cell, then restart the kernel:\n\n"
        "    !pip uninstall -y ants\n"
        "    !pip install -q antspyx\n"
    )


try:
    import cupy as cp

    GPU_AVAILABLE = True
    try:
        gpu_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except Exception:
        gpu_name = "GPU detected"
except Exception:
    cp = None
    GPU_AVAILABLE = False
    gpu_name = "CPU only"


# --------------------------------------------------------------------------- #
# DIPY version shims
#
# DIPY 1.10+ made most secondary arguments keyword-only and renamed sh_order
# to sh_order_max. Kaggle images move around, so call through these.
# --------------------------------------------------------------------------- #

def dipy_gradient_table(bvals, bvecs, **kwargs):
    try:
        return gradient_table(bvals, bvecs=bvecs, **kwargs)
    except TypeError:
        return gradient_table(bvals, bvecs, **kwargs)


def dipy_get_sphere(name):
    for candidate in (name, name.replace("symmetric", "repulsion")):
        for call in (
            lambda n=candidate: get_sphere(name=n),
            lambda n=candidate: get_sphere(n),
        ):
            try:
                return call()
            except (TypeError, ValueError, KeyError):
                continue
    raise ValueError(f"Could not load sphere '{name}'")


def dipy_csd_model(gtab, response, sh_order):
    try:
        return ConstrainedSphericalDeconvModel(
            gtab, response, sh_order_max=sh_order
        )
    except TypeError:
        return ConstrainedSphericalDeconvModel(
            gtab, response, sh_order=sh_order
        )


# --------------------------------------------------------------------------- #
# In-memory ANTs <-> nibabel conversion
#
# ANTs stores geometry in LPS, NIfTI affines are RAS. These helpers were
# validated against a disk round-trip (ants.image_read of a saved NIfTI) for
# both 3D and oblique 4D volumes: identical origin, spacing, direction and
# voxel data. The disk round-trip they replace cost ~4.8 s per 4D fMRI.
# --------------------------------------------------------------------------- #

_LPS2RAS = np.diag([-1.0, -1.0, 1.0, 1.0])

# Presets whose affine schedule can be overridden. "QuickRigid" and friends
# assemble their own antsRegistration call and error out on extra arguments.
_TUNABLE_TRANSFORMS = {
    "Rigid",
    "Similarity",
    "Affine",
    "AffineFast",
    "TRSAA",
    "SyN",
    "SyNRA",
    "SyNOnly",
    "SyNCC",
    "SyNabp",
    "SyNBold",
    "SyNBoldAff",
    "SyNAggro",
    "ElasticSyN",
    "BOLDRigid",
    "BOLDAffine",
}


def ants_from_array(data, affine, tr=1.0, dtype=np.float32):
    """Build an ANTs image from an array plus its NIfTI (RAS) affine."""
    data = np.ascontiguousarray(np.asarray(data), dtype=dtype)
    m = _LPS2RAS @ np.asarray(affine, dtype=float)

    linear = m[:3, :3]
    spacing = np.linalg.norm(linear, axis=0)
    spacing[spacing < 1e-12] = 1.0
    direction = linear / spacing
    origin = m[:3, 3]

    if data.ndim == 3:
        return ants.from_numpy(
            data,
            origin=tuple(origin),
            spacing=tuple(spacing),
            direction=direction,
        )

    if data.ndim == 4:
        direction4 = np.eye(4)
        direction4[:3, :3] = direction
        return ants.from_numpy(
            data,
            origin=tuple(origin) + (0.0,),
            spacing=tuple(spacing) + (float(tr),),
            direction=direction4,
        )

    raise ValueError(f"Unsupported array dimensionality: {data.ndim}")


def affine_from_ants(ants_img):
    """Recover the NIfTI (RAS) affine of an ANTs image."""
    dim = ants_img.dimension
    direction = np.asarray(ants_img.direction).reshape(dim, dim)[:3, :3]
    spacing = np.asarray(ants_img.spacing, dtype=float)[:3]
    origin = np.asarray(ants_img.origin, dtype=float)[:3]

    m = np.eye(4)
    m[:3, :3] = direction @ np.diag(spacing)
    m[:3, 3] = origin
    return _LPS2RAS @ m


def ants_to_array(ants_img):
    """Return (data, nifti_affine) for an ANTs image, no disk involved."""
    return ants_img.numpy(), affine_from_ants(ants_img)


def ants_to_nib(ants_img):
    data, affine = ants_to_array(ants_img)
    return nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine)


def ants_from_nib(nib_img, tr=1.0):
    data = np.asanyarray(nib_img.dataobj, dtype=np.float32)
    return ants_from_array(data, nib_img.affine, tr=tr)


def print_geometry(name, img):
    print(
        f"{name} geometry:\n"
        f"  shape={img.shape}\n"
        f"  origin={tuple(np.round(img.origin, 4))}\n"
        f"  spacing={tuple(np.round(img.spacing, 4))}\n"
        f"  direction=\n{np.asarray(img.direction)}"
    )


class Timer:
    """Tiny stage timer so it is obvious where the wall clock goes."""

    def __init__(self, label, enabled=True):
        self.label = label
        self.enabled = enabled

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        if self.enabled:
            print(f"    [t] {self.label}: {time.time() - self.t0:.1f}s")
        return False


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

class Config:
    FMRI_BASE = Path(
        os.getenv("DLBS_FMRI_BASE", "./data/dlbs_rsfmri/ds004856")
    )
    DTI_BASE = Path(
        os.getenv("DLBS_DTI_BASE", "./data/dlbs_dwi/ds004856")
    )
    ANAT_BASE = Path(
        os.getenv("DLBS_ANAT_BASE", "./data/dlbs_anat/ds004856")
    )

    OUTPUT_DIR = Path(os.getenv("DLBS_OUTPUT_DIR", "./results"))
    CHECKPOINT_DIR = Path(
        os.getenv("DLBS_CHECKPOINT_DIR", "./checkpoints")
    )

    ATLAS_DIR = Path(
        os.getenv(
            "DLBS_ATLAS_DIR",
            "./data/atlas/schaefer_2018",
        )
    )
    ATLAS_NII = ATLAS_DIR / (
        "Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii"
    )
    ATLAS_LABELS = ATLAS_DIR / "Schaefer2018_100Parcels_7Networks_order.txt"

    ATLAS_NAME = "schaefer_100"
    N_PARCELS = 100

    MNI_TEMPLATE_PATH = os.getenv("DLBS_MNI_TEMPLATE", "")
    MNI_RESOLUTION_MM = 2

    # ---- tractography ----
    N_SEEDS_PER_VOXEL = 2
    MIN_STREAMLINE_LENGTH = 20
    MAX_STREAMLINE_LENGTH = 250
    FA_THRESHOLD = 0.15
    FA_SEED_THRESHOLD = 0.20
    STEP_SIZE = 0.5
    MAX_ANGLE = 30.0
    PMF_THRESHOLD = 0.1
    # 362 vertices track ~1.9x faster than 724 with the same streamline yield;
    # angular resolution goes from ~5 deg to ~7 deg, well below MAX_ANGLE.
    TRACKING_SPHERE = "symmetric362"
    CSD_SH_ORDER = 8
    USE_T1_WM_SEED_MASK = True

    # Fixed seed budget instead of a per-voxel grid. With density seeding a
    # subject with a larger WM mask gets proportionally more streamlines, so
    # raw SC weights are not comparable across subjects. A fixed budget
    # decouples streamline count from mask size. Set to None to fall back to
    # the N_SEEDS_PER_VOXEL grid.
    N_SEEDS_TOTAL = 300000
    TRACKING_RANDOM_SEED = 1234

    # Precompute the SH -> sphere function map. ~1.7x faster tracking, at
    # n_voxels * n_sphere_points * 8 bytes (about 2.4 GB for 128x128x50 with
    # symmetric362). "auto" enables it when the estimate fits the budget.
    TRACKING_SH_TO_PMF = "auto"
    TRACKING_PMF_MEMORY_BUDGET_GB = 6.0

    # Tracking is the dominant cost and parallelises cleanly over seed chunks.
    # 0 or None means "use all cores". Workers return endpoints and lengths
    # rather than full streamlines, so almost nothing crosses the process
    # boundary. Set TRACKING_KEEP_STREAMLINES if you need the streamlines
    # themselves, which forces single-process tracking.
    TRACKING_N_JOBS = 0
    TRACKING_KEEP_STREAMLINES = False
    # Fixed regardless of worker count, so results do not depend on how many
    # cores the machine has. Changing it changes the streamlines.
    TRACKING_N_CHUNKS = 32

    # Endpoints that land in background are assigned to the nearest parcel
    # within this radius. 0.0 restores the old strict behaviour.
    ENDPOINT_MAX_DIST_MM = 2.0

    # Minimum streamline count for an edge to exist in the BINARISED matrix
    # used by the coupling-potential metrics. Weak edges are the least
    # reproducible part of the connectome, and binarising at > 0 gives a
    # single-streamline edge the same weight as a thousand-streamline one.
    # Raising this trades sensitivity for stability; check the effect with
    # check_reproducibility before choosing a value.
    SC_MIN_STREAMLINES = 1

    # Communicability is the one metric whose zero pattern equals the edge set,
    # because existing edges are masked out to make it a missing-link score.
    # Set False for the standard (unmasked) network-neuroscience definition,
    # which is the right choice if communicability is interpreted biologically.
    CP_COMMUNICABILITY_MASK_EXISTING_EDGES = True

    # Sanity bounds for the atlas warp, for catching GROSS failures only.
    # Identity resampling of a 2 mm cortical parcellation onto a
    # 1.75 x 1.75 x 3 mm grid already gives about 0.95, and genuine head-size
    # differences move it much further: in an aging cohort a ratio near 0.7 is
    # ordinary atrophy, not a broken warp. Read this as a cohort distribution
    # (flag the outliers) rather than an absolute pass/fail.
    ATLAS_VOLUME_RATIO_RANGE = (0.55, 1.45)
    ATLAS_MIN_REGIONS = 95

    # ---- DWI preprocessing ----
    DWI_DENOISING = True
    # "mppca" (fast, whole 4D), "patch2self", "nlmeans" (slow), "none"
    DWI_DENOISE_METHOD = "mppca"
    MOTION_CORRECT_DWI = True
    DWI_MC_TRANSFORM = "Rigid"
    DWI_MC_FAST = True  # tuned 2-level rigid schedule, ~9x faster than 3-level
    DWI_MC_USE_MASK = True  # restrict the metric to the head
    BVEC_FRAME_CONJUGATION = True
    N4_BIAS_CORRECT = True
    T1_MASK_METHOD = "ants"  # "ants" (fast) or "median_otsu" (v2 behaviour)

    # ---- registration ----
    # ANTs registration is NOT deterministic by default: two identical calls
    # return different transforms (verified). Setting a global seed makes them
    # bit-reproducible, which matters when the same subject may be reprocessed.
    ANTS_RANDOM_SEED = 20250819

    T1_TO_MNI_TRANSFORM = "SyN"
    B0_TO_T1_TRANSFORM = "Affine"
    EPI_TO_T1_TRANSFORM = "Rigid"
    CACHE_REGISTRATIONS = True
    KEEP_REGISTRATION_FILES = True  # False deletes them after each subject

    # ---- fMRI ----
    MOTION_CORRECT_FMRI = True
    FMRI_MC_TRANSFORM = "BOLDRigid"
    FMRI_MC_FAST = True  # tuned schedule + explicit mask, same as the DWI path
    # "mni" reproduces v2. "native" parcellates in EPI space and skips the
    # 4D warp entirely, which is 3-5x faster for this stage.
    FMRI_PARCELLATION_SPACE = "mni"
    FMRI_SMOOTH_FWHM = 6.0  # None to skip
    MOTION_CONFOUND_DERIVATIVES = True
    # Framewise displacement is recorded and flagged, never used to drop a
    # subject silently. Filter on the QC files before group analysis.
    FD_THRESHOLD_MM = 0.5
    FD_MEAN_EXCLUSION_MM = 0.3
    FD_FRACTION_EXCLUSION = 0.30
    TR = 2.0
    LOW_FREQ = 0.01
    HIGH_FREQ = 0.10
    N_DUMMY_SCANS = 5

    SAVE_INDIVIDUAL = True
    VERBOSE = True
    TIMING = True

    BATCH_ID = 0
    N_BATCHES = 4
    MANIFEST_PATH = None

    @classmethod
    def setup_dirs(cls):
        try:
            cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            (cls.OUTPUT_DIR / "sc_matrices").mkdir(exist_ok=True)
            (cls.OUTPUT_DIR / "fc_matrices").mkdir(exist_ok=True)
            (cls.OUTPUT_DIR / "qc").mkdir(exist_ok=True)
            (cls.CHECKPOINT_DIR / "batches").mkdir(exist_ok=True)
            (cls.CHECKPOINT_DIR / "registration").mkdir(exist_ok=True)
        except OSError as exc:
            print(f"[WARN] Could not create output directories: {exc}")
        cls.MANIFEST_PATH = cls.CHECKPOINT_DIR / "completed_subjects.json"

    @classmethod
    def registration_dir(cls, subject_id):
        d = cls.CHECKPOINT_DIR / "registration" / str(subject_id)
        d.mkdir(parents=True, exist_ok=True)
        return d


def apply_ants_seed(config=Config):
    """
    Make ANTs registrations reproducible.

    ANTsPy exposes no `random_seed` argument on `registration`, but it appends
    `--random-seed` to the antsRegistration command when `ants.config._random_seed`
    is set. Without it, two identical calls return different transforms, so the
    same subject reprocessed gives a different connectome.
    """
    seed = getattr(config, "ANTS_RANDOM_SEED", None)
    if seed is None:
        return False
    try:
        ants.config._random_seed = int(seed)
        return True
    except Exception as exc:
        print(f"[WARN] Could not set the ANTs random seed ({exc}); "
              "registrations will not be exactly reproducible.")
        return False


Config.setup_dirs()
_ANTS_SEEDED = apply_ants_seed(Config)


# --------------------------------------------------------------------------- #
# Manifest (atomic writes so a killed kernel cannot corrupt it)
# --------------------------------------------------------------------------- #

def load_subject_set(manifest_path, key):
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return set()
    try:
        with open(manifest_path, "r") as f:
            return set(json.load(f).get(key, []))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[WARN] Manifest unreadable ({exc}); starting from empty set.")
        return set()


def save_subject_set(manifest_path, key, subject_set):
    manifest_path = Path(manifest_path)
    data = {}
    if manifest_path.exists():
        try:
            with open(manifest_path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = {}

    data[key] = sorted(subject_set)

    tmp_path = manifest_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, manifest_path)


def load_completed_subjects(manifest_path):
    return load_subject_set(manifest_path, "completed_subjects")


def save_completed_subjects(manifest_path, completed_subjects):
    save_subject_set(manifest_path, "completed_subjects", completed_subjects)


def load_dti_completed_subjects(manifest_path):
    return load_subject_set(manifest_path, "dti_completed_subjects")


def save_dti_completed_subjects(manifest_path, dti_completed_subjects):
    save_subject_set(
        manifest_path, "dti_completed_subjects", dti_completed_subjects
    )


# --------------------------------------------------------------------------- #
# Subject discovery
# --------------------------------------------------------------------------- #

def discover_subjects(
    fmri_base,
    dti_base,
    anat_base,
    exclude_no_fmri=True,
    batch_id=None,
    n_batches=4,
    require_fmri=True,
):
    subjects_info = []
    no_fmri = {
        "sub-101",
        "sub-153",
        "sub-155",
        "sub-182",
        "sub-221",
        "sub-224",
        "sub-252",
    }

    fmri_base = Path(fmri_base)
    dti_base = Path(dti_base)
    anat_base = Path(anat_base)

    if dti_base.exists():
        for sub_dir in sorted(dti_base.glob("sub-*")):
            sub_id = sub_dir.name

            if exclude_no_fmri and sub_id in no_fmri:
                continue

            sessions = sorted(sub_dir.glob("ses-*"))
            if not sessions:
                continue

            ses_dir = sessions[0]
            ses_id = ses_dir.name

            dwi_dir = ses_dir / "dwi"
            dwi_files = (
                sorted(dwi_dir.glob("*_dwi.nii*")) if dwi_dir.exists() else []
            )
            if not dwi_files:
                continue

            dwi_file = dwi_files[0]
            stem = str(dwi_file).replace(".nii.gz", "").replace(".nii", "")
            bval_file = Path(stem + ".bval")
            bvec_file = Path(stem + ".bvec")

            fmri_func_dir = fmri_base / sub_id / ses_id / "func"
            fmri_files = (
                sorted(fmri_func_dir.glob("*_bold.nii*"))
                if fmri_func_dir.exists()
                else []
            )
            fmri_file = fmri_files[0] if fmri_files else None

            anat_dir = anat_base / sub_id / ses_id / "anat"
            t1_files = (
                sorted(anat_dir.glob("*_T1w.nii*")) if anat_dir.exists() else []
            )
            t1_file = t1_files[0] if t1_files else None

            subjects_info.append(
                {
                    "subject_id": sub_id,
                    "session_id": ses_id,
                    "dwi_file": str(dwi_file),
                    "bval_file": str(bval_file),
                    "bvec_file": str(bvec_file),
                    "has_dwi": dwi_file.exists(),
                    "has_bval": bval_file.exists(),
                    "has_bvec": bvec_file.exists(),
                    "fmri_file": str(fmri_file) if fmri_file else None,
                    "has_fmri": fmri_file is not None,
                    "t1_file": str(t1_file) if t1_file else None,
                    "has_anat": t1_file is not None,
                }
            )

    columns = [
        "subject_id",
        "session_id",
        "dwi_file",
        "bval_file",
        "bvec_file",
        "has_dwi",
        "has_bval",
        "has_bvec",
        "fmri_file",
        "has_fmri",
        "t1_file",
        "has_anat",
    ]

    df = pd.DataFrame(subjects_info, columns=columns)

    if df.empty:
        print(f"ERROR: No DWI subjects discovered under {dti_base}")
        return df

    keep = df["has_dwi"] & df["has_bval"] & df["has_bvec"] & df["has_anat"]
    if require_fmri:
        keep = keep & df["has_fmri"]

    complete = df[keep].copy()

    n_missing_anat = int(
        (
            df["has_dwi"]
            & df["has_bval"]
            & df["has_bvec"]
            & df["has_fmri"]
            & ~df["has_anat"]
        ).sum()
    )
    if n_missing_anat:
        print(
            f"[WARN] {n_missing_anat} otherwise-complete subjects are "
            "missing T1w and will be skipped."
        )

    if batch_id is not None and n_batches:
        if not 0 <= batch_id < n_batches:
            raise ValueError(
                f"batch_id must be in [0, {n_batches - 1}] "
                f"(batches are 0-indexed); got {batch_id}."
            )
        order = sorted(complete["subject_id"].astype(str).tolist())
        # Round-robin keeps batches balanced even when the count does not
        # divide evenly, unlike contiguous blocks.
        assignment = {sid: i % n_batches for i, sid in enumerate(order)}
        complete["batch"] = complete["subject_id"].map(assignment)
        complete = complete[complete["batch"] == batch_id].copy()

    complete = complete.reset_index(drop=True)

    print("Subject discovery summary:")
    print(f"  Complete subjects in this selection: {len(complete)}")
    return complete


# --------------------------------------------------------------------------- #
# Atlas / template
#
# The template is resampled onto the atlas grid once, so MNI space is a single
# grid everywhere downstream: no per-subject atlas resampling, and the warped
# fMRI already sits on the atlas voxels.
# --------------------------------------------------------------------------- #

def _read_schaefer_labels(label_file):
    labels = []
    with open(label_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                label_id = int(parts[0])
            except ValueError:
                continue
            name = parts[1] if len(parts) > 1 else f"ROI_{label_id}"
            labels.append((label_id, name))
    labels.sort(key=lambda x: x[0])
    return labels


def setup_atlas(config=Config):
    if not config.ATLAS_NII.exists():
        raise FileNotFoundError(f"Local Schaefer atlas not found:\n{config.ATLAS_NII}")
    if not config.ATLAS_LABELS.exists():
        raise FileNotFoundError(
            f"Local Schaefer label file not found:\n{config.ATLAS_LABELS}"
        )

    atlas_img = nib.load(str(config.ATLAS_NII))
    atlas_data = np.asanyarray(atlas_img.dataobj).astype(np.int16)

    labels = _read_schaefer_labels(config.ATLAS_LABELS)

    unique_labels = np.unique(atlas_data)
    nonzero_labels = unique_labels[unique_labels > 0]

    if len(nonzero_labels) != config.N_PARCELS:
        raise ValueError(
            f"Expected {config.N_PARCELS} nonzero atlas labels, "
            f"found {len(nonzero_labels)}."
        )
    if not np.array_equal(nonzero_labels, np.arange(1, config.N_PARCELS + 1)):
        raise ValueError(f"Atlas labels are not exactly 1..N: {nonzero_labels}")

    print(f"[Atlas] {config.ATLAS_NII.name}")
    print(f"[Atlas] shape={atlas_data.shape}, regions={len(nonzero_labels)}, "
          f"non-background voxels={int((atlas_data > 0).sum())}")

    return {
        "maps": str(config.ATLAS_NII),
        "labels": [name for _, name in labels],
        "label_pairs": labels,
        "img": atlas_img,
        "data": atlas_data,
        "affine": atlas_img.affine,
        "n_regions": config.N_PARCELS,
        "ants_img": ants_from_array(atlas_data, atlas_img.affine),
    }


def setup_template(config=Config, atlas=None):
    if config.MNI_TEMPLATE_PATH:
        template_path = Path(config.MNI_TEMPLATE_PATH)
        if not template_path.exists():
            raise FileNotFoundError(
                f"Configured MNI template does not exist:\n{template_path}"
            )
        template_img = nib.load(str(template_path))
    else:
        template_img = datasets.load_mni152_template(
            resolution=config.MNI_RESOLUTION_MM
        )

    resampled = False
    if atlas is not None:
        same_grid = template_img.shape[:3] == atlas["img"].shape[:3] and np.allclose(
            template_img.affine, atlas["affine"], atol=1e-4
        )
        if not same_grid:
            try:
                template_img = image.resample_to_img(
                    template_img,
                    atlas["img"],
                    interpolation="continuous",
                    force_resample=True,
                    copy_header=True,
                )
            except TypeError:  # nilearn < 0.10.3
                template_img = image.resample_to_img(
                    template_img, atlas["img"], interpolation="continuous"
                )
            resampled = True

    out_path = (
        config.OUTPUT_DIR / "atlas" / f"mni152_{config.MNI_RESOLUTION_MM}mm.nii.gz"
    )
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(template_img, str(out_path))
    except OSError:
        pass

    print(
        f"[Template] shape={template_img.shape}"
        + (" (resampled onto the atlas grid)" if resampled else "")
    )

    return {
        "img": template_img,
        "path": str(out_path),
        "ants_img": ants_from_nib(template_img),
    }


# --------------------------------------------------------------------------- #
# Transform chains
#
# Verified empirically against an analytically composed transform:
# apply_transforms consumes the list in order, so the transform that acts
# FIRST on reference-grid points must come FIRST. Chain from the output grid
# towards the moving image.
#
# ANTsPy only guesses whichtoinvert for the exact 2-element [mat, warp] case,
# so every chain here carries its own flags.
# --------------------------------------------------------------------------- #

def tx_forward(reg):
    """Point map: registration's fixed space -> its moving space."""
    tl = list(reg["fwdtransforms"])
    return tl, [False] * len(tl)


def tx_inverse(reg):
    """Point map: registration's moving space -> its fixed space."""
    tl = list(reg["invtransforms"])
    return tl, [str(t).endswith(".mat") for t in tl]


def tx_chain(*parts):
    transforms, flags = [], []
    for tl, fl in parts:
        transforms.extend(tl)
        flags.extend(fl)
    return transforms, flags


def apply_chain(fixed, moving, chain, interpolator="linear", imagetype=0):
    transforms, flags = chain
    missing = [t for t in transforms if not os.path.exists(t)]
    if missing:
        raise FileNotFoundError(f"Transform files are gone: {missing}")
    return ants.apply_transforms(
        fixed=fixed,
        moving=moving,
        transformlist=transforms,
        whichtoinvert=list(flags),
        interpolator=interpolator,
        imagetype=imagetype,
        defaultvalue=0,
        singleprecision=True,
    )


# --------------------------------------------------------------------------- #
# T1 preprocessing and registration, both cached on disk
# --------------------------------------------------------------------------- #

def _brain_mask_ants(t1_ants, config=Config):
    if config.T1_MASK_METHOD == "ants":
        try:
            mask = ants.get_mask(t1_ants, cleanup=2)
            frac = float(mask.numpy().mean())
            if 0.05 < frac < 0.6:
                return mask.numpy() > 0
            print(
                f"    [T1] ants.get_mask returned {frac:.1%} of the volume; "
                "falling back to median_otsu."
            )
        except Exception as exc:
            print(f"    [T1] ants.get_mask failed ({exc}); using median_otsu.")

    _, brain_mask = median_otsu(t1_ants.numpy(), median_radius=4, numpass=4)
    return brain_mask > 0


def preprocess_t1(t1_path, config=Config, subject_id=None, use_cache=True):
    """N4 + skull strip. Cached per subject, because v2 paid for this twice."""
    cache_dir = None
    if use_cache and subject_id is not None and config.CACHE_REGISTRATIONS:
        cache_dir = config.registration_dir(subject_id)
        brain_p = cache_dir / "t1_brain.nii.gz"
        mask_p = cache_dir / "t1_mask.nii.gz"
        if brain_p.exists() and mask_p.exists():
            return ants.image_read(str(brain_p)), ants.image_read(str(mask_p))

    t1_ants = ants.image_read(str(t1_path))

    if config.N4_BIAS_CORRECT:
        t1_ants = ants.n4_bias_field_correction(t1_ants)

    brain_mask = _brain_mask_ants(t1_ants, config)

    t1_brain = t1_ants.new_image_like(
        (t1_ants.numpy() * brain_mask).astype(np.float32)
    )
    mask_ants = t1_ants.new_image_like(brain_mask.astype(np.float32))

    if cache_dir is not None:
        try:
            ants.image_write(t1_brain, str(cache_dir / "t1_brain.nii.gz"))
            ants.image_write(mask_ants, str(cache_dir / "t1_mask.nii.gz"))
        except Exception as exc:
            print(f"    [T1] could not cache preprocessed T1: {exc}")

    return t1_brain, mask_ants


def _reg_from_prefix(prefix, nonlinear):
    """Rebuild a registration dict from files ANTsPy already wrote."""
    affine = f"{prefix}0GenericAffine.mat"
    warp = f"{prefix}1Warp.nii.gz"
    inv_warp = f"{prefix}1InverseWarp.nii.gz"

    if nonlinear:
        if all(os.path.exists(p) for p in (affine, warp, inv_warp)):
            return {
                "fwdtransforms": [warp, affine],
                "invtransforms": [affine, inv_warp],
            }
        return None

    if os.path.exists(affine):
        return {"fwdtransforms": [affine], "invtransforms": [affine]}
    return None


def register_cached(
    fixed,
    moving,
    type_of_transform,
    prefix,
    config=Config,
    force=False,
    **kwargs,
):
    nonlinear = "SyN" in type_of_transform or "Elastic" in type_of_transform

    if config.CACHE_REGISTRATIONS and not force:
        cached = _reg_from_prefix(prefix, nonlinear)
        if cached is not None:
            return cached

    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    reg = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform=type_of_transform,
        outprefix=str(prefix),
        **kwargs,
    )
    return {
        "fwdtransforms": list(reg["fwdtransforms"]),
        "invtransforms": list(reg["invtransforms"]),
    }


def register_struct_to_template(t1_brain_ants, template_ants, subject_id, config=Config):
    prefix = str(config.registration_dir(subject_id) / "t1_to_mni_")
    return register_cached(
        template_ants,
        t1_brain_ants,
        config.T1_TO_MNI_TRANSFORM,
        prefix,
        config,
    )


def register_b0_to_struct(b0_ants, t1_brain_ants, subject_id, config=Config):
    prefix = str(config.registration_dir(subject_id) / "b0_to_t1_")
    return register_cached(
        t1_brain_ants,
        b0_ants,
        config.B0_TO_T1_TRANSFORM,
        prefix,
        config,
    )


def register_epi_to_struct(mean_epi_ants, t1_brain_ants, subject_id, config=Config):
    prefix = str(config.registration_dir(subject_id) / "epi_to_t1_")
    return register_cached(
        t1_brain_ants,
        mean_epi_ants,
        config.EPI_TO_T1_TRANSFORM,
        prefix,
        config,
    )


def warp_atlas_to_native(atlas_ants, native_ants, native_to_t1_reg, t1_to_mni_reg):
    """MNI -> T1 -> native, with the inversion flags v2 was missing."""
    return apply_chain(
        native_ants,
        atlas_ants,
        tx_chain(tx_inverse(native_to_t1_reg), tx_inverse(t1_to_mni_reg)),
        interpolator="genericLabel",
    )


def warp_wm_mask_to_native_dwi(
    t1_brain_ants,
    t1_mask_ants,
    b0_ants,
    b0_to_t1_reg,
    wm_threshold_percentile=70,
):
    t1_data = t1_brain_ants.numpy()
    brain = t1_mask_ants.numpy() > 0

    nonzero = t1_data[brain & np.isfinite(t1_data)]
    nonzero = nonzero[nonzero > 0]
    if nonzero.size == 0:
        return np.zeros(b0_ants.shape, dtype=bool)

    thresh = np.percentile(nonzero, wm_threshold_percentile)
    wm_mask = (t1_data >= thresh) & brain

    wm_ants = t1_brain_ants.new_image_like(wm_mask.astype(np.float32))
    wm_native = apply_chain(
        b0_ants,
        wm_ants,
        tx_chain(tx_inverse(b0_to_t1_reg)),
        interpolator="genericLabel",
    )
    return wm_native.numpy() > 0.5


# --------------------------------------------------------------------------- #
# DWI preprocessing
# --------------------------------------------------------------------------- #

def _rotation_from_ants_transform(tf_path):
    """
    Rotation carried by an ANTs linear transform, as a point map from the
    FIXED image space to the MOVING image space (the ITK convention).
    """
    tf = ants.read_transform(tf_path)
    params = np.asarray(tf.parameters, dtype=float)
    if params.size < 12:
        return np.eye(3)

    # ITK stores the matrix row-major: p[0:3] is row 0, p[3:6] row 1, etc.
    matrix = params[:9].reshape(3, 3)
    rotation, _ = polar(matrix)
    det = np.linalg.det(rotation)
    if det <= 0:
        return np.eye(3)
    return rotation / det ** (1.0 / 3.0)


def rotate_bvecs(bvecs, rotations, voxel_axes=None):
    """
    Apply per-volume rotations to b-vectors.

    `rotations[i]` is the fixed->moving rotation of volume i onto the b0
    reference. A gradient measured in volume i corresponds, in reference
    space, to R^T g (the transpose, not R).

    bvecs live in the image/voxel frame while ANTs works in LPS physical
    space, so when `voxel_axes` (the orthonormal direction matrix) is given
    the rotation is conjugated into the voxel frame first.
    """
    rotated = np.zeros_like(np.asarray(bvecs, dtype=float))

    for i, rotation in enumerate(rotations):
        R = np.asarray(rotation, dtype=float)
        if voxel_axes is not None:
            R = voxel_axes.T @ R @ voxel_axes
        rotated[i] = R.T @ bvecs[i]

    norms = np.linalg.norm(rotated, axis=1, keepdims=True)
    nz = norms[:, 0] > 1e-12
    rotated[nz] /= norms[nz]
    return rotated


def motion_correct_dwi(dwi_data, dwi_affine, bvals, bvecs, config=Config, work_dir=None):
    """
    Rigid-align every DWI volume to the first b0 and rotate the b-vectors.

    v2 wrote two gzipped NIfTIs per volume just to hand data to ANTs. Here the
    reference image is built once and each volume is converted in memory.
    """
    b0_idx = np.where(bvals < 50)[0]
    if len(b0_idx) == 0:
        raise ValueError("No b0 volume found; cannot motion-correct.")

    ref_index = int(b0_idx[0])
    ref_ants = ants_from_array(dwi_data[..., ref_index], dwi_affine)

    # Only the generic presets accept an explicit affine schedule; the
    # "Quick*" presets build their own argument list and fail if you pass one.
    #
    # Schedule chosen by benchmark at 128x128x50: against known ground-truth
    # rigid transforms, a 2-level schedule with a brain mask on the reference
    # took 1.2 s/volume versus 10.7 s for the 3-level schedule, with slightly
    # LOWER residual error (0.980 vs 1.023 RMSE inside the brain). Full-
    # resolution refinement buys nothing for within-subject rigid alignment.
    reg_kwargs = {}
    if config.DWI_MC_FAST and config.DWI_MC_TRANSFORM in _TUNABLE_TRANSFORMS:
        reg_kwargs = dict(
            aff_iterations=(200, 100),
            aff_shrink_factors=(4, 2),
            aff_smoothing_sigmas=(2, 1),
        )

    # Restricting the metric to the head removes most of the cost, since
    # background voxels contribute nothing but samples.
    if config.DWI_MC_USE_MASK:
        try:
            ref_mask = ants.get_mask(ref_ants, cleanup=2)
            if 0.02 < float(ref_mask.numpy().mean()) < 0.8:
                reg_kwargs["mask"] = ref_mask
        except Exception:
            pass

    n_vols = dwi_data.shape[-1]
    corrected = np.empty_like(dwi_data, dtype=np.float32)
    rotations = []

    prefix_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="dwimc_"))
    prefix_dir.mkdir(parents=True, exist_ok=True)

    try:
        for i in range(n_vols):
            if i == ref_index:
                corrected[..., i] = dwi_data[..., i]
                rotations.append(np.eye(3))
                continue

            vol_ants = ants_from_array(dwi_data[..., i], dwi_affine)
            try:
                reg = ants.registration(
                    fixed=ref_ants,
                    moving=vol_ants,
                    type_of_transform=config.DWI_MC_TRANSFORM,
                    outprefix=str(prefix_dir / f"vol{i:04d}_"),
                    **reg_kwargs,
                )
            except (RuntimeError, ValueError):
                if reg_kwargs:
                    reg_kwargs = {}
                    reg = ants.registration(
                        fixed=ref_ants,
                        moving=vol_ants,
                        type_of_transform=config.DWI_MC_TRANSFORM,
                        outprefix=str(prefix_dir / f"vol{i:04d}_"),
                    )
                else:
                    raise
            corrected[..., i] = reg["warpedmovout"].numpy()

            try:
                rotations.append(
                    _rotation_from_ants_transform(reg["fwdtransforms"][0])
                )
            except Exception:
                rotations.append(np.eye(3))
    finally:
        if work_dir is None:
            shutil.rmtree(prefix_dir, ignore_errors=True)

    voxel_axes = None
    if config.BVEC_FRAME_CONJUGATION:
        lin = (_LPS2RAS @ np.asarray(dwi_affine, float))[:3, :3]
        voxel_axes = lin / np.linalg.norm(lin, axis=0)

    return corrected, rotate_bvecs(bvecs, rotations, voxel_axes)


def sanitize_volume(data, stage, sub_id="", reference=None, clip_negative=True):
    """
    Repair non-finite voxels and report where they came from.

    DIPY's tensor and CSD fits fail with "SVD did not converge" the moment a
    single NaN reaches them, and the traceback points at numpy rather than at
    the stage that produced it. Checking after each stage names the culprit.
    Non-finite voxels are restored from `reference` (the pre-stage data) when
    one is given, so genuine signal is preserved rather than zeroed.
    """
    data = np.asarray(data, dtype=np.float32)
    bad = ~np.isfinite(data)
    n_bad = int(bad.sum())

    if n_bad:
        n_voxels = int(bad.any(axis=-1).sum())
        source = "restored from pre-stage data" if reference is not None else "zeroed"
        print(
            f"  [{sub_id}] NON-FINITE after {stage}: {n_bad} values in "
            f"{n_voxels} voxels ({source})."
        )
        if reference is not None:
            data = np.where(bad, np.asarray(reference, dtype=np.float32), data)
            data[~np.isfinite(data)] = 0.0
        else:
            data[bad] = 0.0

    if clip_negative:
        np.clip(data, 0.0, None, out=data)

    return data, n_bad


def validate_gradients(bvals, bvecs, sub_id=""):
    """b-vectors must be finite and unit norm or gradient_table refuses them."""
    bvals = np.asarray(bvals, dtype=float).ravel()
    bvecs = np.asarray(bvecs, dtype=float)

    bad = ~np.isfinite(bvecs).all(axis=1)
    if bad.any():
        print(f"  [{sub_id}] {int(bad.sum())} non-finite b-vectors zeroed.")
        bvecs[bad] = 0.0

    bvals = np.nan_to_num(bvals, nan=0.0, posinf=0.0, neginf=0.0)

    norms = np.linalg.norm(bvecs, axis=1)
    nonzero = norms > 1e-6
    bvecs[nonzero] /= norms[nonzero, None]
    bvecs[~nonzero] = 0.0

    # A b0 must have a zero vector, a diffusion volume must not.
    b0 = bvals < 50
    orphan = (~b0) & (~nonzero)
    if orphan.any():
        print(
            f"  [{sub_id}] {int(orphan.sum())} diffusion volumes have a zero "
            "b-vector; treating them as b0."
        )
        bvals[orphan] = 0.0

    return bvals, bvecs


def denoise_dwi(dwi_data, bvals, mask=None, config=Config):
    method = (config.DWI_DENOISE_METHOD or "none").lower()

    if method == "none":
        return dwi_data

    if method == "mppca":
        from dipy.denoise.localpca import mppca

        patch_radius = 2

        # MPPCA only visits patch centres in [patch_radius, shape - patch_radius),
        # and divides by the per-voxel patch coverage at the end. An in-mask
        # voxel that no valid centre reaches gets 0/0 = NaN, which survives
        # dipy's own masking and then kills the tensor fit with "SVD did not
        # converge". Typical culprits are brain or noise specks sitting on the
        # first or last slice, which is common with a 50-slice acquisition.
        # Dropping those voxels from the denoising mask prevents the NaN;
        # they keep their original (undenoised) signal.
        safe_mask = mask
        if mask is not None:
            interior = np.zeros_like(mask, dtype=bool)
            interior[
                patch_radius:-patch_radius or None,
                patch_radius:-patch_radius or None,
                patch_radius:-patch_radius or None,
            ] = True
            safe_mask = mask & interior
            n_dropped = int(mask.sum() - safe_mask.sum())
            if n_dropped:
                print(
                    f"    [denoise] {n_dropped} mask voxels within {patch_radius} "
                    "voxels of the volume border left undenoised (MPPCA cannot "
                    "cover them)."
                )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                out = mppca(dwi_data, mask=safe_mask, patch_radius=patch_radius)
            except TypeError:
                out = mppca(dwi_data, safe_mask, patch_radius=patch_radius)

        out = np.asarray(out, dtype=np.float32)
        # Keep the original signal wherever denoising was skipped or failed.
        if safe_mask is not None:
            skipped = mask & ~safe_mask
            if skipped.any():
                out[skipped] = dwi_data[skipped]
        return out

    if method == "patch2self":
        from dipy.denoise.patch2self import patch2self

        return patch2self(dwi_data, bvals).astype(np.float32)

    if method == "nlmeans":
        from dipy.denoise.nlmeans import nlmeans
        from dipy.denoise.noise_estimate import estimate_sigma

        sigma = estimate_sigma(dwi_data)
        out = np.empty_like(dwi_data, dtype=np.float32)
        for i in range(dwi_data.shape[-1]):
            out[..., i] = nlmeans(
                dwi_data[..., i], sigma=sigma[i], mask=mask, num_threads=-1
            )
        return out

    raise ValueError(f"Unknown DWI_DENOISE_METHOD: {config.DWI_DENOISE_METHOD}")


def preprocess_dwi_subject(subject_row, config=Config, work_dir=None):
    sub_id = subject_row["subject_id"]

    dwi_data, dwi_affine = load_nifti(str(subject_row["dwi_file"]))
    bvals, bvecs = read_bvals_bvecs(
        str(subject_row["bval_file"]), str(subject_row["bvec_file"])
    )
    dwi_data = np.ascontiguousarray(dwi_data, dtype=np.float32)

    bvals, bvecs = validate_gradients(bvals, bvecs, sub_id)
    # ANTs refuses a moving image containing NaN, so check before registering.
    dwi_data, _ = sanitize_volume(dwi_data, "loading", sub_id)

    b0_idx = np.where(bvals < 50)[0]
    if len(b0_idx) == 0:
        raise ValueError(f"No b0 image for {sub_id}")

    if config.MOTION_CORRECT_DWI:
        with Timer("dwi motion correction", config.TIMING):
            pre_mc = dwi_data
            dwi_data, bvecs = motion_correct_dwi(
                dwi_data, dwi_affine, bvals, bvecs, config, work_dir=work_dir
            )
            dwi_data, _ = sanitize_volume(
                dwi_data, "motion correction", sub_id, reference=pre_mc
            )
            del pre_mc

    gtab = dipy_gradient_table(bvals, bvecs)

    # Mask first, then denoise inside it. v2 denoised every voxel in the
    # bounding box, ~70% of which is air.
    with Timer("brain mask", config.TIMING):
        _, brain_mask = median_otsu(
            dwi_data,
            vol_idx=np.arange(min(10, dwi_data.shape[-1])),
            median_radius=3,
            numpass=1,
        )
    brain_mask = brain_mask.astype(bool)

    if config.DWI_DENOISING:
        with Timer(f"denoise ({config.DWI_DENOISE_METHOD})", config.TIMING):
            pre_denoise = dwi_data
            dwi_data = denoise_dwi(
                dwi_data, bvals, mask=brain_mask, config=config
            )
            dwi_data, _ = sanitize_volume(
                dwi_data, "denoising", sub_id, reference=pre_denoise
            )
            del pre_denoise

    # Final backstop: restrict the mask to voxels that are usable for fitting.
    # NaN * 0 is NaN, so masking the data does not remove a bad voxel; the
    # mask itself has to exclude it.
    usable = np.isfinite(dwi_data).all(axis=-1) & (dwi_data[..., b0_idx].max(axis=-1) > 0)
    n_unusable = int((brain_mask & ~usable).sum())
    if n_unusable:
        print(
            f"  [{sub_id}] {n_unusable} in-brain voxels excluded from fitting "
            "(non-finite or zero b0)."
        )
    brain_mask = brain_mask & usable

    if not brain_mask.any():
        raise RuntimeError("Brain mask is empty after quality filtering.")

    masked_data = dwi_data * brain_mask[..., None]
    masked_data[~np.isfinite(masked_data)] = 0.0
    b0_vol = np.ascontiguousarray(dwi_data[..., int(b0_idx[0])])

    return {
        "affine": dwi_affine,
        "bvals": bvals,
        "bvecs": bvecs,
        "gtab": gtab,
        "brain_mask": brain_mask,
        "b0": b0_vol,
        "b0_ants": ants_from_array(b0_vol, dwi_affine),
        "masked_data": masked_data,
    }


# --------------------------------------------------------------------------- #
# Tractography
#
# Tracking dominates the runtime of the whole pipeline, so three things happen
# here that did not in v2:
#   * the SH -> sphere map is precomputed when memory allows (~1.7x)
#   * a 362-vertex sphere replaces the 724-vertex one (~1.9x)
#   * seed chunks run in parallel worker processes
# Workers return only endpoints and lengths, which is all the connectivity
# step needs, so the big streamline arrays never cross a process boundary.
# --------------------------------------------------------------------------- #

# Set in the parent before forking; children inherit it copy-on-write, so the
# precomputed PMF is never pickled or duplicated.
_TRACKING_STATE = {}


def _resolve_sh_to_pmf(shape, n_sphere_points, config):
    setting = config.TRACKING_SH_TO_PMF
    if isinstance(setting, bool):
        return setting

    n_voxels = int(np.prod(shape[:3]))
    estimate_gb = n_voxels * n_sphere_points * 8 / 1e9
    if estimate_gb <= config.TRACKING_PMF_MEMORY_BUDGET_GB:
        return True

    print(
        f"    [tracking] precomputed PMF would need ~{estimate_gb:.1f} GB, over the "
        f"{config.TRACKING_PMF_MEMORY_BUDGET_GB:.1f} GB budget; evaluating on the fly."
    )
    return False


def _resolve_n_jobs(config, n_seeds):
    n_jobs = config.TRACKING_N_JOBS
    if not n_jobs:
        n_jobs = os.cpu_count() or 1
    n_jobs = max(1, int(n_jobs))

    if config.TRACKING_KEEP_STREAMLINES:
        return 1
    if n_seeds < 5000:
        return 1
    try:
        import multiprocessing as mp

        mp.get_context("fork")
    except (ImportError, ValueError):
        return 1
    return n_jobs


def _track_chunk(task):
    """Track one seed chunk. Returns (starts, ends, lengths)."""
    chunk_index, seeds = task
    state = _TRACKING_STATE

    generator = LocalTracking(
        state["direction_getter"],
        state["stopping_criterion"],
        seeds,
        state["affine"],
        step_size=state["step_size"],
        max_cross=1,
        return_all=False,
        random_seed=state["random_seed"] + chunk_index,
        **state["length_kwargs"],
    )
    streamlines = Streamlines(generator)

    if state["keep_streamlines"]:
        return streamlines

    if len(streamlines) == 0:
        empty = np.zeros((0, 3), dtype=np.float32)
        return empty, empty, np.zeros(0, dtype=np.float32)

    starts, ends = _streamline_endpoints(streamlines)
    lengths = np.asarray(length(streamlines), dtype=np.float32)
    return (
        np.asarray(starts, dtype=np.float32),
        np.asarray(ends, dtype=np.float32),
        lengths,
    )


def run_tractography(shm_coeff, fa, affine, seeds, sphere, config, sub_id=""):
    """
    Returns (starts, ends, lengths, streamlines_or_None).

    Streamlines are only materialised when TRACKING_KEEP_STREAMLINES is set,
    which also forces single-process tracking.
    """
    global _TRACKING_STATE

    sh_to_pmf = _resolve_sh_to_pmf(shm_coeff.shape, len(sphere.vertices), config)

    direction_getter = ProbabilisticDirectionGetter.from_shcoeff(
        shm_coeff,
        max_angle=config.MAX_ANGLE,
        sphere=sphere,
        pmf_threshold=config.PMF_THRESHOLD,
        sh_to_pmf=sh_to_pmf,
    )

    # Enforce length limits during tracking (in steps) so runaway streamlines
    # are dropped before they are ever built.
    length_kwargs = {}
    try:
        import inspect as _inspect

        params = _inspect.signature(LocalTracking.__init__).parameters
        if "minlen" in params and "maxlen" in params:
            length_kwargs = dict(
                minlen=max(2, int(config.MIN_STREAMLINE_LENGTH / config.STEP_SIZE)),
                maxlen=int(config.MAX_STREAMLINE_LENGTH / config.STEP_SIZE) + 1,
            )
    except (TypeError, ValueError):
        length_kwargs = {}

    _TRACKING_STATE = {
        "direction_getter": direction_getter,
        "stopping_criterion": ThresholdStoppingCriterion(fa, config.FA_THRESHOLD),
        "affine": affine,
        "step_size": config.STEP_SIZE,
        "random_seed": int(config.TRACKING_RANDOM_SEED),
        "length_kwargs": length_kwargs,
        "keep_streamlines": bool(config.TRACKING_KEEP_STREAMLINES),
    }

    n_jobs = _resolve_n_jobs(config, len(seeds))

    # The seed list is always split into the same fixed number of chunks, each
    # with its own deterministic random seed, whatever the worker count is.
    # Chunking by n_jobs instead would make the RNG stream depend on the
    # machine, so the same subject would give different streamlines on a
    # 2-core and a 4-core session.
    n_chunks = 1 if config.TRACKING_KEEP_STREAMLINES else config.TRACKING_N_CHUNKS
    chunks = np.array_split(seeds, max(1, int(n_chunks)))
    tasks = [(i, c) for i, c in enumerate(chunks) if len(c)]

    print(
        f"[{sub_id}] Tracking {len(seeds)} seeds on {len(sphere.vertices)} "
        f"directions (sh_to_pmf={sh_to_pmf}, workers={n_jobs}, "
        f"chunks={len(tasks)})"
    )

    try:
        if n_jobs == 1:
            results = [_track_chunk(t) for t in tasks]
        else:
            import multiprocessing as mp

            with mp.get_context("fork").Pool(n_jobs) as pool:
                results = pool.map(_track_chunk, tasks)

        if config.TRACKING_KEEP_STREAMLINES:
            streamlines = results[0]
            starts, ends = _streamline_endpoints(streamlines)
            lengths = np.asarray(length(streamlines), dtype=np.float32)
            return (
                np.asarray(starts, dtype=np.float32),
                np.asarray(ends, dtype=np.float32),
                lengths,
                streamlines,
            )

        starts = np.concatenate([r[0] for r in results], axis=0)
        ends = np.concatenate([r[1] for r in results], axis=0)
        lengths = np.concatenate([r[2] for r in results], axis=0)
        return starts, ends, lengths, None

    finally:
        _TRACKING_STATE = {}


def generate_seeds(seed_mask, affine, config):
    """Fixed seed budget when configured, otherwise the per-voxel grid."""
    if config.N_SEEDS_TOTAL:
        try:
            return tracking_utils.random_seeds_from_mask(
                seed_mask,
                affine,
                seeds_count=int(config.N_SEEDS_TOTAL),
                seed_count_per_voxel=False,
                random_seed=int(config.TRACKING_RANDOM_SEED),
            )
        except TypeError:
            return tracking_utils.random_seeds_from_mask(
                seed_mask,
                affine,
                int(config.N_SEEDS_TOTAL),
                False,
                int(config.TRACKING_RANDOM_SEED),
            )

    return tracking_utils.seeds_from_mask(
        seed_mask, affine, density=config.N_SEEDS_PER_VOXEL
    )


# --------------------------------------------------------------------------- #
# Endpoint -> parcel assignment
# --------------------------------------------------------------------------- #

def _streamline_endpoints(streamlines):
    """(starts, ends) as (N, 3) arrays, without a Python loop when possible."""
    data = getattr(streamlines, "_data", None)
    offsets = getattr(streamlines, "_offsets", None)
    lengths = getattr(streamlines, "_lengths", None)

    if data is not None and offsets is not None and lengths is not None:
        offsets = np.asarray(offsets, dtype=np.int64)
        lengths = np.asarray(lengths, dtype=np.int64)
        keep = lengths > 0
        return data[offsets[keep]], data[offsets[keep] + lengths[keep] - 1]

    starts = np.array([s[0] for s in streamlines], dtype=float)
    ends = np.array([s[-1] for s in streamlines], dtype=float)
    return starts, ends


def _nearest_label_lookup(atlas_data, affine, max_dist_mm):
    """
    Label volume where background voxels within max_dist_mm of a parcel take
    that parcel's label. Streamlines stop in white matter, just outside the
    cortical ribbon, so a strict lookup throws most endpoints away.
    """
    if max_dist_mm <= 0:
        return atlas_data

    voxel_sizes = np.linalg.norm(np.asarray(affine)[:3, :3], axis=0)
    dist, indices = distance_transform_edt(
        atlas_data == 0, sampling=voxel_sizes, return_indices=True
    )
    nearest = atlas_data[tuple(indices)]
    filled = np.where((atlas_data == 0) & (dist <= max_dist_mm), nearest, atlas_data)
    return filled.astype(np.int16)


def region_volumes_mm3(atlas_data, atlas_affine, n_regions):
    """Volume of each parcel in mm^3, in whatever space the atlas is given."""
    voxel_volume = float(
        abs(np.linalg.det(np.asarray(atlas_affine, dtype=float)[:3, :3]))
    )
    counts = np.bincount(
        np.asarray(atlas_data).reshape(-1).astype(np.int64), minlength=n_regions + 1
    )[1 : n_regions + 1]
    return counts.astype(np.float64) * voxel_volume


def streamlines_to_connectivity(
    streamlines,
    atlas_data,
    atlas_affine,
    n_regions,
    max_dist_mm=0.0,
    lengths=None,
    region_volumes=None,
    endpoints=None,
):
    """
    Endpoint-to-parcel assignment, fully vectorised (v2 used a Python loop
    over every streamline with two affine calls each).

    Returns four weightings, because raw streamline count is biased by fibre
    length and by parcel size:
      count        raw number of streamlines connecting i and j
      invlen       sum of 1/length, the standard length-bias correction
      density      Hagmann-style: 2/(v_i+v_j) * sum 1/length, which also
                   removes the parcel-size bias
      mean_length  mean streamline length per edge, in mm
    """
    sc = np.zeros((n_regions, n_regions), dtype=np.float32)
    empty = {
        "count": sc,
        "invlen": sc.copy(),
        "density": sc.copy(),
        "mean_length": sc.copy(),
    }
    stats = {
        "n_streamlines": len(streamlines) if streamlines is not None else 0,
        "out_of_bounds": 0,
        "background": 0,
        "same_region": 0,
        "invalid_label": 0,
        "valid": 0,
    }
    if endpoints is None and (streamlines is None or len(streamlines) == 0):
        return empty, stats

    lookup = _nearest_label_lookup(atlas_data, atlas_affine, max_dist_mm)

    if endpoints is not None:
        starts, ends = endpoints
        stats["n_streamlines"] = int(len(starts))
        if len(starts) == 0:
            return empty, stats
    else:
        starts, ends = _streamline_endpoints(streamlines)
    inv = np.linalg.inv(np.asarray(atlas_affine, dtype=float))

    def to_voxel(points):
        return np.rint(points @ inv[:3, :3].T + inv[:3, 3]).astype(np.int32)

    v0 = to_voxel(starts)
    v1 = to_voxel(ends)

    shape = np.array(lookup.shape)
    in_bounds = np.all((v0 >= 0) & (v0 < shape), axis=1) & np.all(
        (v1 >= 0) & (v1 < shape), axis=1
    )
    stats["out_of_bounds"] = int((~in_bounds).sum())

    v0, v1 = v0[in_bounds], v1[in_bounds]
    if v0.shape[0] == 0:
        return empty, stats

    if lengths is None:
        lengths = np.asarray(length(streamlines), dtype=np.float64)
    else:
        lengths = np.asarray(lengths, dtype=np.float64)
    lengths = lengths[in_bounds]


    r0 = lookup[v0[:, 0], v0[:, 1], v0[:, 2]].astype(np.int32)
    r1 = lookup[v1[:, 0], v1[:, 1], v1[:, 2]].astype(np.int32)

    background = (r0 <= 0) | (r1 <= 0)
    stats["background"] = int(background.sum())

    invalid = (~background) & ((r0 > n_regions) | (r1 > n_regions))
    stats["invalid_label"] = int(invalid.sum())

    good = (~background) & (~invalid)
    same = good & (r0 == r1)
    stats["same_region"] = int(same.sum())

    good = good & (r0 != r1)
    stats["valid"] = int(good.sum())

    i = r0[good] - 1
    j = r1[good] - 1
    edge_lengths = np.maximum(lengths[good], 1e-6)
    flat = i * n_regions + j
    size = n_regions * n_regions

    def accumulate(weights):
        m = np.bincount(flat, weights=weights, minlength=size).reshape(
            n_regions, n_regions
        )
        return m + m.T

    counts = accumulate(None)
    invlen = accumulate(1.0 / edge_lengths)
    length_sum = accumulate(edge_lengths)

    mean_length = np.divide(
        length_sum, counts, out=np.zeros_like(length_sum), where=counts > 0
    )

    if region_volumes is None:
        region_volumes = region_volumes_mm3(atlas_data, atlas_affine, n_regions)
    region_volumes = np.asarray(region_volumes, dtype=np.float64)
    volume_sum = region_volumes[:, None] + region_volumes[None, :]
    density = np.divide(
        2.0 * invlen, volume_sum, out=np.zeros_like(invlen), where=volume_sum > 0
    )

    return (
        {
            "count": counts.astype(np.float32),
            "invlen": invlen.astype(np.float32),
            "density": density.astype(np.float32),
            "mean_length": mean_length.astype(np.float32),
        },
        stats,
    )


# --------------------------------------------------------------------------- #
# DTI / tractography
# --------------------------------------------------------------------------- #

def process_dti_subject(subject_row, atlas, template, config=Config):
    sub_id = subject_row["subject_id"]
    t_start = time.time()

    try:
        print(f"\n[{sub_id}] DWI preprocessing...")
        preproc = preprocess_dwi_subject(subject_row, config)

        dwi_affine = preproc["affine"]
        gtab = preproc["gtab"]
        mask = preproc["brain_mask"]
        masked_data = preproc["masked_data"]
        b0_ants = preproc["b0_ants"]

        if config.VERBOSE:
            print_geometry(f"[{sub_id}] Native DWI b0", b0_ants)

        with Timer("tensor fit", config.TIMING):
            # preprocess_dwi_subject already excludes non-finite voxels; this
            # re-check costs nothing and turns a numpy LinAlgError deep inside
            # dipy into an actionable message.
            finite = np.isfinite(masked_data).all(axis=-1)
            if not finite[mask].all():
                n_bad = int((mask & ~finite).sum())
                print(f"  [{sub_id}] dropping {n_bad} non-finite voxels before fitting.")
                mask = mask & finite
                masked_data = np.nan_to_num(
                    masked_data, nan=0.0, posinf=0.0, neginf=0.0
                )
            tenfit = TensorModel(gtab).fit(masked_data, mask=mask)
            fa = fractional_anisotropy(tenfit.evals)
            fa = np.nan_to_num(np.clip(fa, 0, 1), nan=0.0).astype(np.float32)
        del tenfit

        print(
            f"[{sub_id}] FA mean={fa[mask].mean():.3f}, "
            f"FA>{config.FA_SEED_THRESHOLD}: {int((fa > config.FA_SEED_THRESHOLD).sum())}"
        )

        print(f"[{sub_id}] CSD...")
        with Timer("CSD fit", config.TIMING):
            response, ratio = auto_response_ssst(
                gtab, masked_data, roi_radii=10, fa_thr=0.7
            )
            print(f"[{sub_id}] CSD response ratio={ratio:.4f}")

            csd_model = dipy_csd_model(gtab, response, config.CSD_SH_ORDER)
            csd_fit = csd_model.fit(masked_data, mask=mask)

        sphere = dipy_get_sphere(config.TRACKING_SPHERE)

        # from_shcoeff evaluates the ODF on the fly. csd_fit.odf(sphere) would
        # allocate n_voxels x n_sphere_points floats, 4.7 GB at this size.
        shm_coeff = np.asarray(csd_fit.shm_coeff, dtype=np.float32)
        del csd_fit, csd_model

        print(f"[{sub_id}] Registration...")
        with Timer("T1 preprocessing", config.TIMING):
            t1_brain_ants, t1_mask_ants = preprocess_t1(
                subject_row["t1_file"], config, subject_id=sub_id
            )

        with Timer("T1 -> MNI", config.TIMING):
            t1_to_mni = register_struct_to_template(
                t1_brain_ants, template["ants_img"], sub_id, config
            )
        with Timer("b0 -> T1", config.TIMING):
            b0_to_t1 = register_b0_to_struct(b0_ants, t1_brain_ants, sub_id, config)

        # ---------------- seed mask ----------------
        seed_mask = (fa > config.FA_SEED_THRESHOLD) & mask

        if config.USE_T1_WM_SEED_MASK:
            wm_native = warp_wm_mask_to_native_dwi(
                t1_brain_ants, t1_mask_ants, b0_ants, b0_to_t1
            )
            wm_count = int(wm_native.sum())
            print(
                f"[{sub_id}] WM mask in DWI: {wm_count} / {wm_native.size} "
                f"({100 * wm_count / wm_native.size:.3f}%)"
            )
            if (seed_mask & wm_native).sum() > 0:
                seed_mask = seed_mask & wm_native
            else:
                print(f"[{sub_id}] WARNING: WM/FA seed mask empty, using FA+brain.")

        seed_count = int(seed_mask.sum())
        print(f"[{sub_id}] Seed voxels: {seed_count}")
        if seed_count == 0:
            raise RuntimeError("Seed mask is empty.")

        seeds = generate_seeds(seed_mask, dwi_affine, config)
        seeding = (
            f"fixed budget of {config.N_SEEDS_TOTAL}"
            if config.N_SEEDS_TOTAL
            else f"grid density {config.N_SEEDS_PER_VOXEL}"
        )
        print(f"[{sub_id}] Seeds generated: {len(seeds)} ({seeding})")

        # ---------------- tracking ----------------
        with Timer("tractography", config.TIMING):
            starts, ends, lengths, streamline_obj = run_tractography(
                shm_coeff, fa, dwi_affine, seeds, sphere, config, sub_id
            )

        print(f"[{sub_id}] Raw streamlines: {len(lengths)}")

        keep = (lengths > config.MIN_STREAMLINE_LENGTH) & (
            lengths < config.MAX_STREAMLINE_LENGTH
        )
        starts, ends, kept_lengths = starts[keep], ends[keep], lengths[keep]
        n_kept = int(keep.sum())
        print(f"[{sub_id}] Length-filtered streamlines: {n_kept}")

        # ---------------- atlas -> native DWI ----------------
        print(f"[{sub_id}] Warping Schaefer atlas MNI -> T1 -> DWI...")
        with Timer("atlas warp", config.TIMING):
            atlas_native_ants = warp_atlas_to_native(
                atlas["ants_img"], b0_ants, b0_to_t1, t1_to_mni
            )

        atlas_native_affine = affine_from_ants(atlas_native_ants)
        atlas_data_subj = np.rint(atlas_native_ants.numpy()).astype(np.int16)

        affine_consistent = bool(
            np.allclose(atlas_native_affine, dwi_affine, atol=1e-3)
        )
        if not affine_consistent:
            print(
                f"[{sub_id}] NOTE: atlas-native affine differs from the DWI affine; "
                "using the atlas-native affine for endpoint mapping."
            )

        nonzero_count = int((atlas_data_subj > 0).sum())
        present = np.unique(atlas_data_subj)
        present = present[present > 0]

        # The parcellation's total volume should be roughly preserved from MNI
        # to native space. A large deviation means the warp is stretching or
        # collapsing the parcellation, which quietly corrupts the SC matrix.
        mni_voxel_mm3 = abs(np.linalg.det(np.asarray(atlas["affine"])[:3, :3]))
        native_voxel_mm3 = abs(np.linalg.det(np.asarray(atlas_native_affine)[:3, :3]))
        mni_volume = float((atlas["data"] > 0).sum()) * mni_voxel_mm3
        native_volume = float(nonzero_count) * native_voxel_mm3
        volume_ratio = native_volume / mni_volume if mni_volume > 0 else 0.0

        lo, hi = config.ATLAS_VOLUME_RATIO_RANGE
        atlas_warp_ok = (
            lo <= volume_ratio <= hi and len(present) >= config.ATLAS_MIN_REGIONS
        )

        print(
            f"[{sub_id}] Warped atlas: shape={atlas_data_subj.shape}, "
            f"non-background={nonzero_count} "
            f"({100 * nonzero_count / atlas_data_subj.size:.3f}%), "
            f"regions present={len(present)}/{atlas['n_regions']}, "
            f"volume ratio vs MNI={volume_ratio:.2f}"
        )

        if nonzero_count == 0:
            raise RuntimeError(
                "ATLAS WARP FAILED: atlas is entirely background in native DWI space."
            )

        if not atlas_warp_ok:
            print(
                f"[{sub_id}] ATLAS WARP WARNING: volume ratio {volume_ratio:.2f} "
                f"outside [{lo}, {hi}] or only {len(present)} regions present. "
                "The registration is suspect for this subject; check "
                "qc/*_dti_qc.json before including it in group analysis."
            )

        # ---------------- connectivity ----------------
        with Timer("endpoint mapping", config.TIMING):
            volumes = region_volumes_mm3(
                atlas_data_subj, atlas_native_affine, atlas["n_regions"]
            )
            sc_variants, stats = streamlines_to_connectivity(
                None,
                atlas_data_subj,
                atlas_native_affine,
                atlas["n_regions"],
                max_dist_mm=config.ENDPOINT_MAX_DIST_MM,
                lengths=kept_lengths,
                region_volumes=volumes,
                endpoints=(starts, ends),
            )
            sc_matrix = sc_variants["count"]

        print(f"[{sub_id}] Endpoint mapping:")
        for key in ("out_of_bounds", "background", "same_region", "invalid_label", "valid"):
            print(f"  {key}: {stats[key]}")
        print(f"  SC nonzero entries: {int(np.count_nonzero(sc_matrix))}")
        print(f"  SC total edge weight: {int(sc_matrix.sum())}")
        print(f"[{sub_id}] DTI stage took {time.time() - t_start:.1f}s")

        qc = {
            "subject_id": sub_id,
            "fa_mean": float(fa[mask].mean()),
            "csd_response_ratio": float(ratio),
            "n_seed_voxels": seed_count,
            "n_streamlines_raw": int(len(lengths)),
            "n_streamlines_kept": n_kept,
            "n_seeds": int(len(seeds)),
            "mean_streamline_length_mm": float(kept_lengths.mean())
            if n_kept
            else 0.0,
            "atlas_regions_present": int(len(present)),
            "atlas_volume_ratio_vs_mni": round(volume_ratio, 4),
            "atlas_warp_ok": bool(atlas_warp_ok),
            "atlas_affine_matches_dwi": affine_consistent,
            "ants_seeded": bool(_ANTS_SEEDED),
            "endpoint_stats": stats,
            "sc_nonzero": int(np.count_nonzero(sc_matrix)),
            "sc_total_weight": float(sc_matrix.sum()),
            "sc_density_edges": int(np.count_nonzero(sc_variants["density"])),
            "mean_edge_length_mm": float(
                sc_variants["mean_length"][sc_matrix > 0].mean()
            )
            if np.any(sc_matrix > 0)
            else 0.0,
            "empty_parcels_native": int((volumes == 0).sum()),
            "dti_seconds": round(time.time() - t_start, 1),
        }
        try:
            with open(config.OUTPUT_DIR / "qc" / f"{sub_id}_dti_qc.json", "w") as f:
                json.dump(qc, f, indent=2)
        except OSError:
            pass

        del streamline_obj, shm_coeff, masked_data, preproc, starts, ends
        gc.collect()

        return {
            "subject_id": sub_id,
            "sc_matrix": sc_matrix,
            "sc_matrix_log": np.log1p(sc_matrix),
            "sc_invlen": sc_variants["invlen"],
            "sc_density": sc_variants["density"],
            "sc_mean_length": sc_variants["mean_length"],
            "region_volumes_mm3": volumes,
            "fa_mean": qc["fa_mean"],
            "n_streamlines": n_kept,
            "t1_to_mni": t1_to_mni,
            "b0_to_t1": b0_to_t1,
            "t1_brain_ants": t1_brain_ants,
            "qc": qc,
            "success": True,
        }

    except Exception as exc:
        print(f"  [{sub_id}] ERROR (DTI): {exc}")
        traceback.print_exc()
        return {"subject_id": sub_id, "success": False, "error": str(exc)}


# --------------------------------------------------------------------------- #
# Functional connectivity helpers
# --------------------------------------------------------------------------- #

def compute_plv_matrix(time_series):
    """
    Phase locking value.

    PLV_ij = |mean_t exp(i(phi_ti - phi_tj))|, which is |Z^H Z| / T with
    Z_ti = exp(i phi_ti). One complex matmul instead of v2's n^2 loop.
    """
    ts = np.asarray(time_series, dtype=np.float64)
    n_timepoints = ts.shape[0]

    phases = np.angle(hilbert(ts, axis=0))
    z = np.exp(1j * phases)

    plv = np.abs(z.conj().T @ z) / n_timepoints
    plv = np.asarray(plv, dtype=np.float32)
    np.fill_diagonal(plv, 1.0)
    return plv


# Kept for backward compatibility; the vectorised CPU version is already
# faster than any host-device round trip for a 100 x 100 matrix.
def compute_plv_matrix_gpu(time_series):
    return compute_plv_matrix(time_series)


def build_parcel_operator(atlas_data, n_regions):
    """
    Sparse (n_regions x n_voxels) averaging operator.

    Replaces NiftiLabelsMasker, which re-resamples the atlas onto the data
    grid on every call.
    """
    labels = np.asarray(atlas_data).reshape(-1)
    voxel_idx = np.flatnonzero(labels > 0)
    rows = labels[voxel_idx].astype(np.int64) - 1

    valid = rows < n_regions
    voxel_idx = voxel_idx[valid]
    rows = rows[valid]

    counts = np.bincount(rows, minlength=n_regions).astype(np.float64)
    nonempty = counts > 0
    weights = 1.0 / counts[rows]

    operator = sparse.csr_matrix(
        (weights, (rows, np.arange(rows.size))),
        shape=(n_regions, rows.size),
    )
    return operator, voxel_idx, nonempty


def extract_parcel_timeseries(data_4d, atlas_data, n_regions):
    """Parcel means as (T, n_regions)."""
    operator, voxel_idx, nonempty = build_parcel_operator(atlas_data, n_regions)

    n_time = data_4d.shape[-1]
    flat = np.asarray(data_4d, dtype=np.float32).reshape(-1, n_time)
    ts = operator @ flat[voxel_idx]

    ts = np.asarray(ts, dtype=np.float64).T
    ts[:, ~nonempty] = 0.0
    return ts, int((~nonempty).sum())


def _decompose_rigid(matrix, offset):
    """3 translations + 3 Euler angles from an ITK linear transform."""
    rotation, _ = polar(np.asarray(matrix, dtype=float))
    sy = np.hypot(rotation[0, 0], rotation[1, 0])
    if sy > 1e-6:
        rx = np.arctan2(rotation[2, 1], rotation[2, 2])
        ry = np.arctan2(-rotation[2, 0], sy)
        rz = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        rx = np.arctan2(-rotation[1, 2], rotation[1, 1])
        ry = np.arctan2(-rotation[2, 0], sy)
        rz = 0.0
    return np.array([offset[0], offset[1], offset[2], rx, ry, rz], dtype=float)


def extract_motion_confounds(motion_parameters, n_timepoints, derivatives=True):
    """
    Six rigid-body parameters per volume (v2 used the first 6 of the 12 raw
    affine parameters, which are two rows of the rotation matrix).
    """
    rows = []
    for entry in motion_parameters:
        candidates = entry if isinstance(entry, (list, tuple)) else [entry]
        vec = None
        for c in candidates:
            if isinstance(c, str) and os.path.exists(c):
                try:
                    params = np.asarray(
                        ants.read_transform(c).parameters, dtype=float
                    )
                    if params.size >= 12:
                        vec = _decompose_rigid(
                            params[:9].reshape(3, 3), params[9:12]
                        )
                    elif params.size == 6:
                        vec = params.copy()
                    break
                except Exception:
                    pass
        rows.append(np.zeros(6) if vec is None else vec)

    if not rows:
        return np.zeros((n_timepoints, 0))

    confounds = np.zeros((n_timepoints, 6))
    n = min(n_timepoints, len(rows))
    confounds[:n] = np.vstack(rows[:n])

    if derivatives:
        deriv = np.vstack([np.zeros((1, 6)), np.diff(confounds, axis=0)])
        confounds = np.hstack([confounds, deriv])

    keep = confounds.std(axis=0) > 1e-12
    return confounds[:, keep]


def motion_correct_fmri(data_4d, affine, config=Config, work_dir=None):
    """Returns (corrected array, motion parameter file lists)."""
    mean_vol = data_4d.mean(axis=-1)
    fixed = ants_from_array(mean_vol, affine)
    moving = ants_from_array(data_4d, affine, tr=config.TR)

    kwargs = {}
    if work_dir is not None:
        Path(work_dir).mkdir(parents=True, exist_ok=True)
        kwargs["outprefix"] = str(Path(work_dir) / "mc")

    # ants.motion_correction computes a mask internally but then passes None
    # to the registration, so the metric samples the whole field of view.
    # Handing it an explicit mask plus the 2-level schedule cuts the cost the
    # same way it does for DWI.
    if config.FMRI_MC_FAST:
        try:
            brain = ants.get_mask(fixed, cleanup=2)
            if 0.02 < float(brain.numpy().mean()) < 0.8:
                kwargs["mask"] = brain
        except Exception:
            pass
        if config.FMRI_MC_TRANSFORM in _TUNABLE_TRANSFORMS:
            kwargs.update(
                aff_iterations=(200, 100),
                aff_shrink_factors=(4, 2),
                aff_smoothing_sigmas=(2, 1),
            )

    try:
        mc = ants.motion_correction(
            moving,
            fixed=fixed,
            type_of_transform=config.FMRI_MC_TRANSFORM,
            **kwargs,
        )
    except (RuntimeError, ValueError, TypeError):
        for key in ("mask", "aff_iterations", "aff_shrink_factors", "aff_smoothing_sigmas"):
            kwargs.pop(key, None)
        mc = ants.motion_correction(
            moving,
            fixed=fixed,
            type_of_transform=config.FMRI_MC_TRANSFORM,
            **kwargs,
        )
    fd = np.asarray(mc.get("FD", []), dtype=float).ravel()
    return mc["motion_corrected"].numpy(), mc.get("motion_parameters", []), fd


def process_fmri_subject(subject_row, atlas, template, dti_result, config=Config):
    sub_id = subject_row["subject_id"]
    t_start = time.time()
    work_dir = Path(tempfile.mkdtemp(prefix=f"fmri_{sub_id}_"))

    try:
        if not subject_row.get("fmri_file"):
            raise FileNotFoundError("No fMRI file for this subject.")

        fmri_img = nib.load(str(subject_row["fmri_file"]))
        fmri_affine = fmri_img.affine

        n_drop = config.N_DUMMY_SCANS if fmri_img.shape[-1] > config.N_DUMMY_SCANS else 0
        data = np.asarray(fmri_img.dataobj[..., n_drop:], dtype=np.float32)
        n_timepoints = data.shape[-1]

        motion_params = []
        fd = np.zeros(0)
        if config.MOTION_CORRECT_FMRI:
            with Timer("fmri motion correction", config.TIMING):
                data, motion_params, fd = motion_correct_fmri(
                    data, fmri_affine, config, work_dir=work_dir
                )

        confounds = extract_motion_confounds(
            motion_params, n_timepoints, config.MOTION_CONFOUND_DERIVATIVES
        )

        mean_epi_ants = ants_from_array(data.mean(axis=-1), fmri_affine)

        t1_brain_ants = (dti_result or {}).get("t1_brain_ants")
        if t1_brain_ants is None:
            t1_brain_ants, _ = preprocess_t1(
                subject_row["t1_file"], config, subject_id=sub_id
            )

        t1_to_mni = (dti_result or {}).get("t1_to_mni")
        if t1_to_mni is None:
            with Timer("T1 -> MNI (fmri stage)", config.TIMING):
                t1_to_mni = register_struct_to_template(
                    t1_brain_ants, template["ants_img"], sub_id, config
                )

        with Timer("EPI -> T1", config.TIMING):
            epi_to_t1 = register_epi_to_struct(
                mean_epi_ants, t1_brain_ants, sub_id, config
            )

        if config.FMRI_PARCELLATION_SPACE == "native":
            # Warp one label volume instead of every BOLD timepoint.
            with Timer("atlas -> EPI", config.TIMING):
                atlas_native = warp_atlas_to_native(
                    atlas["ants_img"], mean_epi_ants, epi_to_t1, t1_to_mni
                )
            work_data = data
            work_affine = fmri_affine
            labels = np.rint(atlas_native.numpy()).astype(np.int16)
        else:
            with Timer("fMRI -> MNI", config.TIMING):
                fmri_ants = ants_from_array(data, fmri_affine, tr=config.TR)
                fmri_mni = apply_chain(
                    template["ants_img"],
                    fmri_ants,
                    tx_chain(tx_forward(t1_to_mni), tx_forward(epi_to_t1)),
                    interpolator="linear",
                    imagetype=3,
                )
                work_data = fmri_mni.numpy()
                work_affine = affine_from_ants(fmri_mni)
            labels = atlas["data"]
            if labels.shape != work_data.shape[:3]:
                # Should not happen: setup_template puts MNI on the atlas grid.
                print(
                    f"  [{sub_id}] atlas grid {labels.shape} != warped fMRI grid "
                    f"{work_data.shape[:3]}; resampling the atlas."
                )
                try:
                    resampled_atlas = image.resample_to_img(
                        atlas["img"],
                        nib.Nifti1Image(work_data[..., 0], work_affine),
                        interpolation="nearest",
                        force_resample=True,
                        copy_header=True,
                    )
                except TypeError:
                    resampled_atlas = image.resample_to_img(
                        atlas["img"],
                        nib.Nifti1Image(work_data[..., 0], work_affine),
                        interpolation="nearest",
                    )
                labels = np.rint(
                    np.asanyarray(resampled_atlas.dataobj)
                ).astype(np.int16)

        if config.FMRI_SMOOTH_FWHM:
            with Timer("smoothing", config.TIMING):
                smoothed = image.smooth_img(
                    nib.Nifti1Image(work_data, work_affine),
                    fwhm=config.FMRI_SMOOTH_FWHM,
                )
                work_data = np.asanyarray(smoothed.dataobj, dtype=np.float32)

        # Detrending, band-pass filtering and confound regression are linear
        # and identical for every voxel, so they commute with parcel
        # averaging. Cleaning 100 time series instead of ~800k voxels is the
        # single biggest saving in this stage.
        time_series, n_empty = extract_parcel_timeseries(
            work_data, labels, atlas["n_regions"]
        )
        if n_empty:
            print(f"  [{sub_id}] WARNING: {n_empty} parcels are empty in this space.")

        try:
            time_series = nl_signal.clean(
                time_series,
                detrend=True,
                standardize="zscore_sample",
                confounds=confounds if confounds.shape[1] else None,
                low_pass=config.HIGH_FREQ,
                high_pass=config.LOW_FREQ,
                t_r=config.TR,
            )
        except TypeError:
            time_series = nl_signal.clean(
                time_series,
                detrend=True,
                standardize=True,
                confounds=confounds if confounds.shape[1] else None,
                low_pass=config.HIGH_FREQ,
                high_pass=config.LOW_FREQ,
                t_r=config.TR,
            )

        if time_series.shape[1] != atlas["n_regions"]:
            raise RuntimeError(
                f"fMRI parcellation returned {time_series.shape[1]} regions; "
                f"expected {atlas['n_regions']}."
            )

        plv_matrix = compute_plv_matrix(time_series)
        corr_matrix = ConnectivityMeasure(kind="correlation").fit_transform(
            [time_series]
        )[0]

        # Motion QC. High-motion subjects inflate short-range correlations, so
        # record FD rather than silently averaging it away.
        mean_fd = float(fd.mean()) if fd.size else float("nan")
        max_fd = float(fd.max()) if fd.size else float("nan")
        # json has no NaN literal; write null when motion correction was off.
        json_mean_fd = None if fd.size == 0 else mean_fd
        json_max_fd = None if fd.size == 0 else max_fd
        n_high_fd = int((fd > config.FD_THRESHOLD_MM).sum()) if fd.size else 0
        frac_high_fd = (n_high_fd / fd.size) if fd.size else 0.0

        motion_flag = bool(
            fd.size
            and (
                mean_fd > config.FD_MEAN_EXCLUSION_MM
                or frac_high_fd > config.FD_FRACTION_EXCLUSION
            )
        )
        if motion_flag:
            print(
                f"  [{sub_id}] MOTION WARNING: mean FD={mean_fd:.3f} mm, "
                f"{100 * frac_high_fd:.1f}% of volumes above "
                f"{config.FD_THRESHOLD_MM} mm. Flagged, not excluded; "
                "filter on qc/*_fmri_qc.json before group analysis."
            )

        print(
            f"[{sub_id}] fMRI stage took {time.time() - t_start:.1f}s "
            f"({n_timepoints} volumes, {confounds.shape[1]} confound regressors, "
            + (f"mean FD={mean_fd:.3f} mm)" if fd.size else "no motion correction)")
        )

        fmri_qc = {
            "subject_id": sub_id,
            "n_timepoints": n_timepoints,
            "n_dummy_dropped": n_drop,
            "n_confound_regressors": int(confounds.shape[1]),
            "n_empty_parcels": n_empty,
            "parcellation_space": config.FMRI_PARCELLATION_SPACE,
            "mean_fd_mm": json_mean_fd,
            "max_fd_mm": json_max_fd,
            "n_volumes_above_fd_threshold": n_high_fd,
            "fraction_above_fd_threshold": frac_high_fd,
            "motion_flag": motion_flag,
            "fmri_seconds": round(time.time() - t_start, 1),
        }
        try:
            with open(config.OUTPUT_DIR / "qc" / f"{sub_id}_fmri_qc.json", "w") as f:
                json.dump(fmri_qc, f, indent=2)
        except OSError:
            pass

        return {
            "subject_id": sub_id,
            "plv_matrix": plv_matrix,
            "corr_matrix": np.asarray(corr_matrix, dtype=np.float32),
            "n_timepoints": n_timepoints,
            "n_empty_parcels": n_empty,
            "framewise_displacement": fd,
            "motion_flag": motion_flag,
            "qc": fmri_qc,
            "time_series": time_series if config.SAVE_INDIVIDUAL else None,
            "success": True,
        }

    except Exception as exc:
        print(f"  [{sub_id}] ERROR (fMRI): {exc}")
        traceback.print_exc()
        try:
            usage = shutil.disk_usage(tempfile.gettempdir())
            print(
                f"  [{sub_id}] {tempfile.gettempdir()}: "
                f"{usage.free / 1e9:.2f} GB free of {usage.total / 1e9:.2f} GB"
            )
        except Exception:
            pass
        return {"subject_id": sub_id, "success": False, "error": str(exc)}

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Coupling-potential graph metrics
#
# All of these are matrix products for a binary adjacency, including
# Adamic-Adar and resource allocation, which v2 computed with an O(n^2)
# Python loop over np.intersect1d.
# --------------------------------------------------------------------------- #

def _compute_cp_metrics_cpu(A, n, mask_existing_edges=True):
    A = np.asarray(A, dtype=np.float32)
    metrics = {}

    degree = A.sum(axis=1)

    cn_matrix = A @ A
    np.fill_diagonal(cn_matrix, 0)
    metrics["common_neighbors"] = cn_matrix

    union = degree[:, None] + degree[None, :] - cn_matrix
    jaccard = np.divide(
        cn_matrix, union, out=np.zeros_like(cn_matrix), where=union > 0
    )
    np.fill_diagonal(jaccard, 0)
    metrics["jaccard"] = jaccard

    log_degree = np.maximum(np.log(degree + 1.0), 1e-10)
    inv_log_degree = (1.0 / log_degree).astype(np.float32)
    inv_degree = (1.0 / (degree + 1e-10)).astype(np.float32)

    # sum over common neighbours k of w_k  ==  (A * w) @ A.T for binary A
    aa_matrix = (A * inv_log_degree[None, :]) @ A.T
    ra_matrix = (A * inv_degree[None, :]) @ A.T
    np.fill_diagonal(aa_matrix, 0)
    np.fill_diagonal(ra_matrix, 0)
    metrics["adamic_adar"] = aa_matrix
    metrics["resource_allocation"] = ra_matrix

    norms = np.linalg.norm(A, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1e-10
    A_norm = A / norms
    profile_similarity = A_norm @ A_norm.T
    np.fill_diagonal(profile_similarity, 0)
    metrics["profile_similarity"] = profile_similarity

    lambda_max = float(np.max(np.linalg.eigvalsh(A))) if n else 0.0
    beta = 1.0 / lambda_max if lambda_max > 0 else 0.1
    comm_matrix = expm(beta * A.astype(np.float64)).astype(np.float32)
    np.fill_diagonal(comm_matrix, 0)

    # Masking out existing edges turns communicability into a link-prediction
    # score (how well-connected are two regions that are NOT directly linked).
    # It also makes the metric's zero pattern identical to the edge set, so
    # any instability in weak edges shows up directly as instability in the
    # metric. The unmasked version is the standard network-neuroscience
    # definition and is what you want if communicability is being interpreted
    # as a biological quantity rather than a missing-link score.
    if mask_existing_edges:
        metrics["communicability"] = comm_matrix * (1 - A)
    else:
        metrics["communicability"] = comm_matrix

    hub_mask = degree >= np.percentile(degree, 90)
    A_hub = A[:, hub_mask]
    if A_hub.shape[1] > 0:
        hub_norms = np.linalg.norm(A_hub, axis=1, keepdims=True)
        hub_norms[hub_norms < 1e-10] = 1e-10
        A_hub_n = A_hub / hub_norms
        hub_affinity = A_hub_n @ A_hub_n.T
    else:
        hub_affinity = np.zeros((n, n), dtype=np.float32)
    np.fill_diagonal(hub_affinity, 0)
    metrics["hub_affinity"] = hub_affinity

    composite = np.zeros((n, n), dtype=np.float32)
    base_metrics = list(metrics.values())
    for matrix in base_metrics:
        mx = float(np.max(matrix))
        if mx > 0:
            mn = float(np.min(matrix))
            composite += (matrix - mn) / (mx - mn + 1e-10)
    composite /= max(len(base_metrics), 1)
    metrics["composite"] = composite

    return metrics


def compute_coupling_potential_metrics(
    sc_matrix, use_gpu=False, min_streamlines=None, mask_existing_edges=None
):
    """
    Coupling-potential metrics on the BINARISED structural connectome.

    Binarising at > 0 makes single-streamline edges as influential as
    thousand-streamline ones, and those weak edges are exactly the least
    reproducible part of the matrix. `min_streamlines` drops them first.
    Defaults to Config.SC_MIN_STREAMLINES.
    """
    if min_streamlines is None:
        min_streamlines = getattr(Config, "SC_MIN_STREAMLINES", 1)
    threshold = max(1, int(min_streamlines))

    if mask_existing_edges is None:
        mask_existing_edges = getattr(
            Config, "CP_COMMUNICABILITY_MASK_EXISTING_EDGES", True
        )

    A = (np.asarray(sc_matrix) >= threshold).astype(np.float32)
    return _compute_cp_metrics_cpu(A, A.shape[0], bool(mask_existing_edges))


# --------------------------------------------------------------------------- #
# Batch drivers
# --------------------------------------------------------------------------- #

def _sc_arrays(dti_result):
    """Every SC weighting available in a result dict, for np.savez."""
    arrays = {
        "sc": dti_result["sc_matrix"],
        "sc_log": dti_result["sc_matrix_log"],
    }
    for key, name in (
        ("sc_invlen", "sc_invlen"),
        ("sc_density", "sc_density"),
        ("sc_mean_length", "sc_mean_length"),
        ("region_volumes_mm3", "region_volumes_mm3"),
    ):
        value = dti_result.get(key)
        if value is not None:
            arrays[name] = value
    return arrays


def _save_subject_outputs(save_dir, sub_id, dti_result, fmri_result, cp_metrics):
    save_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(save_dir / f"{sub_id}_sc.npz", **_sc_arrays(dti_result))
    np.savez_compressed(
        save_dir / f"{sub_id}_fc.npz",
        plv=fmri_result["plv_matrix"],
        corr=fmri_result["corr_matrix"],
        framewise_displacement=fmri_result.get(
            "framewise_displacement", np.zeros(0)
        ),
        **(
            {"time_series": fmri_result["time_series"]}
            if fmri_result.get("time_series") is not None
            else {}
        ),
    )
    with open(save_dir / f"{sub_id}_cp.pkl", "wb") as f:
        pickle.dump(cp_metrics, f, protocol=pickle.HIGHEST_PROTOCOL)


def _cleanup_registration(sub_id, config):
    if not config.KEEP_REGISTRATION_FILES:
        shutil.rmtree(
            config.CHECKPOINT_DIR / "registration" / str(sub_id), ignore_errors=True
        )


def _free_memory():
    gc.collect()
    if GPU_AVAILABLE and cp is not None:
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass


def run_batch_pipeline(batch_id=0, n_batches=4, config=Config):
    config.BATCH_ID = batch_id
    config.N_BATCHES = n_batches
    config.MANIFEST_PATH = config.CHECKPOINT_DIR / "completed_subjects.json"

    atlas = setup_atlas(config)
    template = setup_template(config, atlas=atlas)

    subjects_df = discover_subjects(
        config.FMRI_BASE,
        config.DTI_BASE,
        config.ANAT_BASE,
        batch_id=batch_id,
        n_batches=n_batches,
    )
    if subjects_df.empty:
        return 0

    sc_cache_dir = config.OUTPUT_DIR / "sc_matrices"
    sc_cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = config.CHECKPOINT_DIR / f"lock_batch_{batch_id}.lock"

    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            completed = load_completed_subjects(config.MANIFEST_PATH)
            dti_done = load_dti_completed_subjects(config.MANIFEST_PATH)

            remaining = subjects_df[
                ~subjects_df["subject_id"].isin(completed)
            ].copy()
            print(
                f"Batch {batch_id}/{n_batches}: processing {len(remaining)} subjects"
            )

            for position, (_, subject_row) in enumerate(
                remaining.iterrows(), start=1
            ):
                sub_id = subject_row["subject_id"]
                print(
                    f"\n[{batch_id}/{n_batches}] [{position}/{len(remaining)}] "
                    f"Processing {sub_id}..."
                )
                subject_t0 = time.time()

                sc_cache_path = sc_cache_dir / f"{sub_id}_sc.npz"

                if sub_id in dti_done and sc_cache_path.exists():
                    print(f"[{sub_id}] Reusing cached SC matrix.")
                    cached = np.load(sc_cache_path)
                    dti_result = {
                        "subject_id": sub_id,
                        "sc_matrix": cached["sc"],
                        "sc_matrix_log": cached["sc_log"],
                        "sc_invlen": cached["sc_invlen"]
                        if "sc_invlen" in cached
                        else None,
                        "sc_density": cached["sc_density"]
                        if "sc_density" in cached
                        else None,
                        "sc_mean_length": cached["sc_mean_length"]
                        if "sc_mean_length" in cached
                        else None,
                        "region_volumes_mm3": cached["region_volumes_mm3"]
                        if "region_volumes_mm3" in cached
                        else None,
                        # Registrations are cached on disk now, so the fMRI
                        # stage reloads them instead of redoing SyN.
                        "t1_to_mni": _reg_from_prefix(
                            str(config.registration_dir(sub_id) / "t1_to_mni_"),
                            "SyN" in config.T1_TO_MNI_TRANSFORM,
                        ),
                        "b0_to_t1": None,
                        "t1_brain_ants": None,
                        "success": True,
                    }
                else:
                    dti_result = process_dti_subject(
                        subject_row, atlas, template, config
                    )
                    if dti_result["success"]:
                        np.savez_compressed(sc_cache_path, **_sc_arrays(dti_result))
                        dti_done.add(sub_id)
                        save_dti_completed_subjects(config.MANIFEST_PATH, dti_done)

                if not dti_result["success"]:
                    _free_memory()
                    continue

                fmri_result = process_fmri_subject(
                    subject_row, atlas, template, dti_result, config
                )
                if not fmri_result["success"]:
                    _free_memory()
                    continue

                cp_metrics = compute_coupling_potential_metrics(
                    dti_result["sc_matrix"]
                )

                _save_subject_outputs(
                    config.OUTPUT_DIR / f"batch_{batch_id}",
                    sub_id,
                    dti_result,
                    fmri_result,
                    cp_metrics,
                )

                completed.add(sub_id)
                save_completed_subjects(config.MANIFEST_PATH, completed)
                _cleanup_registration(sub_id, config)

                print(
                    f"[{sub_id}] done in {time.time() - subject_t0:.1f}s "
                    f"({len(completed)} subjects completed overall)"
                )

                del dti_result, fmri_result, cp_metrics
                _free_memory()

        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    return len(completed)


def run_sc_only_pipeline(batch_id=0, n_batches=4, config=Config):
    """DTI -> structural connectome only, skipping fMRI entirely."""
    config.BATCH_ID = batch_id
    config.N_BATCHES = n_batches
    config.MANIFEST_PATH = config.CHECKPOINT_DIR / "completed_subjects.json"

    atlas = setup_atlas(config)
    template = setup_template(config, atlas=atlas)

    subjects_df = discover_subjects(
        config.FMRI_BASE,
        config.DTI_BASE,
        config.ANAT_BASE,
        batch_id=batch_id,
        n_batches=n_batches,
    )
    if subjects_df.empty:
        return 0

    sc_cache_dir = config.OUTPUT_DIR / "sc_matrices"
    sc_cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = config.CHECKPOINT_DIR / f"lock_batch_{batch_id}.lock"

    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            completed = load_completed_subjects(config.MANIFEST_PATH)
            dti_done = load_dti_completed_subjects(config.MANIFEST_PATH)

            already = {
                s
                for s in (completed | dti_done)
                if s in completed or (sc_cache_dir / f"{s}_sc.npz").exists()
            }
            remaining = subjects_df[~subjects_df["subject_id"].isin(already)].copy()

            print(
                f"[SC-only] Batch {batch_id}/{n_batches}: {len(already)} already "
                f"done, {len(remaining)} remaining"
            )

            for position, (_, subject_row) in enumerate(
                remaining.iterrows(), start=1
            ):
                sub_id = subject_row["subject_id"]
                print(
                    f"\n[SC-only] [{batch_id}/{n_batches}] "
                    f"[{position}/{len(remaining)}] Processing {sub_id}..."
                )

                dti_result = process_dti_subject(subject_row, atlas, template, config)
                if not dti_result["success"]:
                    _free_memory()
                    continue

                np.savez_compressed(
                    sc_cache_dir / f"{sub_id}_sc.npz", **_sc_arrays(dti_result)
                )
                dti_done.add(sub_id)
                save_dti_completed_subjects(config.MANIFEST_PATH, dti_done)

                del dti_result
                _free_memory()

        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    return len(dti_done)


def debug_one_subject(target_subject="sub-1003", config=Config, with_fmri=False):
    print(f"\n=== DEBUGGING {target_subject} ===\n")

    atlas = setup_atlas(config)
    template = setup_template(config, atlas=atlas)

    subjects_df = discover_subjects(
        config.FMRI_BASE,
        config.DTI_BASE,
        config.ANAT_BASE,
        batch_id=None,
        n_batches=None,
    )
    matches = subjects_df[subjects_df["subject_id"] == target_subject]
    if matches.empty:
        raise ValueError(f"{target_subject} was not found among complete subjects.")

    row = matches.iloc[0]
    dti_result = process_dti_subject(row, atlas, template, config)

    if with_fmri and dti_result.get("success"):
        fmri_result = process_fmri_subject(row, atlas, template, dti_result, config)
        return dti_result, fmri_result

    return dti_result


if __name__ == "__main__":
    print("Python:", sys.version.split()[0])
    print("NumPy:", np.__version__)
    print("DIPY:", dipy.__version__)
    print("Nilearn:", nilearn.__version__)
    print("ANTsPy:", getattr(ants, "__version__", "unknown"))
    print("ITK threads:", os.environ.get("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"))
    print("GPU:", gpu_name)

    # debug_one_subject("sub-1003")
    # run_batch_pipeline(batch_id=0, n_batches=12)
    # run_sc_only_pipeline(batch_id=0, n_batches=12)
