
import os
import gc
import time
import pickle
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import expm

import nibabel as nib
import nilearn
from nilearn import image
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure

from dipy.io.image import load_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel, fractional_anisotropy
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking.streamline import Streamlines
from dipy.tracking import utils as tracking_utils
from dipy.direction import peaks_from_model
from dipy.data import default_sphere
from dipy.segment.mask import median_otsu

warnings.filterwarnings("ignore")
np.random.seed(42)

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

class Config:
    FMRI_BASE = Path(os.environ.get("DLBS_FMRI_DIR", "/kaggle/input/dlbs-raw-rs-fmri-dataset/dlbs_rsfmri/ds004856"))
    DTI_BASE = Path(os.environ.get("DLBS_DTI_DIR", "/kaggle/input/dlbs-dti-raw-data/dlbs_dwi/ds004856"))
    OUTPUT_DIR = Path(os.environ.get("DLBS_OUTPUT_DIR", "./results"))
    CHECKPOINT_DIR = Path(os.environ.get("DLBS_CHECKPOINT_DIR", "./checkpoints"))
    ATLAS_NAME = "schaefer_100"
    N_PARCELS = 100
    N_SUBJECTS = int(os.environ.get("DLBS_N_SUBJECTS", "193"))
    N_SEEDS_PER_VOXEL = 1
    MIN_STREAMLINE_LENGTH = 20
    MAX_STREAMLINE_LENGTH = 250
    FA_THRESHOLD = 0.15
    TR = 2.0
    LOW_FREQ = 0.01
    HIGH_FREQ = 0.1
    N_DUMMY_SCANS = 5
    SAVE_INDIVIDUAL = True
    VERBOSE = True

    @classmethod
    def setup_dirs(cls):
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        for name in ("sc_matrices", "fc_matrices", "figures"):
            (cls.OUTPUT_DIR / name).mkdir(exist_ok=True)

Config.setup_dirs()

def discover_subjects(fmri_base, dti_base, exclude_no_fmri=True):
    subjects_info = []
    
    no_fmri = {'sub-101', 'sub-153', 'sub-155', 'sub-182', 'sub-221', 'sub-224', 'sub-252'}
    
    if dti_base.exists():
        for sub_dir in sorted(dti_base.glob("sub-*")):
            sub_id = sub_dir.name
            
            if exclude_no_fmri and sub_id in no_fmri:
                continue
            
            sessions = list(sub_dir.glob("ses-*"))
            if not sessions:
                continue
            ses_dir = sessions[0]
            ses_id = ses_dir.name
            
            dwi_dir = ses_dir / "dwi"
            dwi_files = list(dwi_dir.glob("*_dwi.nii*")) if dwi_dir.exists() else []
            
            if not dwi_files:
                continue
            
            dwi_file = dwi_files[0]
            bval_file = dwi_file.with_suffix('').with_suffix('.bval') if '.gz' in str(dwi_file) else dwi_file.with_suffix('.bval')
            bvec_file = dwi_file.with_suffix('').with_suffix('.bvec') if '.gz' in str(dwi_file) else dwi_file.with_suffix('.bvec')
            
            stem = str(dwi_file).replace('.nii.gz', '').replace('.nii', '')
            bval_file = Path(stem + '.bval')
            bvec_file = Path(stem + '.bvec')
            
            fmri_func_dir = fmri_base / sub_id / ses_id / "func"
            fmri_files = list(fmri_func_dir.glob("*_bold.nii*")) if fmri_func_dir.exists() else []
            
            has_fmri = len(fmri_files) > 0
            fmri_file = fmri_files[0] if has_fmri else None
            
            subjects_info.append({
                'subject_id': sub_id,
                'session_id': ses_id,
                'dwi_file': str(dwi_file),
                'bval_file': str(bval_file),
                'bvec_file': str(bvec_file),
                'has_dwi': dwi_file.exists(),
                'has_bval': bval_file.exists(),
                'has_bvec': bvec_file.exists(),
                'fmri_file': str(fmri_file) if fmri_file else None,
                'has_fmri': has_fmri
            })
    
    df = pd.DataFrame(subjects_info)
    
    complete = df[df['has_dwi'] & df['has_bval'] & df['has_bvec'] & df['has_fmri']]
    
    print(f"Subject Discovery Summary:")
    print(f"  Total subjects found: {len(df)}")
    print(f"  With complete DTI: {df['has_dwi'].sum()}")
    print(f"  With complete fMRI: {df['has_fmri'].sum()}")
    print(f"  Complete cases (both modalities): {len(complete)}")
    
    return complete.reset_index(drop=True)

subjects_df = discover_subjects(Config.FMRI_BASE, Config.DTI_BASE)

if len(subjects_df) > Config.N_SUBJECTS:
    subjects_df = subjects_df.sample(n=Config.N_SUBJECTS, random_state=42).reset_index(drop=True)
    print(f"\n→ Selected {Config.N_SUBJECTS} subjects for processing")

subjects_df.head(10)


def setup_atlas():
    from nilearn.datasets import fetch_atlas_schaefer_2018
    
    atlas = fetch_atlas_schaefer_2018(
        n_rois=Config.N_PARCELS,
        yeo_networks=7,
        resolution_mm=2,
        data_dir=str(Config.OUTPUT_DIR / "atlas")
    )
    
    atlas_img = nib.load(atlas.maps)
    atlas_data = atlas_img.get_fdata()
    atlas_affine = atlas_img.affine
    
    labels = atlas.labels
    
    print(f"✓ Atlas loaded: {Config.ATLAS_NAME}")
    print(f"  Shape: {atlas_data.shape}")
    print(f"  N regions: {len(np.unique(atlas_data)) - 1}")  # Exclude 0 (background)
    print(f"  Labels: {labels[:5]}... (showing first 5)")
    
    return {
        'maps': atlas.maps,
        'labels': labels,
        'img': atlas_img,
        'data': atlas_data,
        'affine': atlas_affine,
        'n_regions': Config.N_PARCELS
    }

atlas = setup_atlas()


def process_dti_subject(subject_row, atlas, config=Config):
    sub_id = subject_row['subject_id']
    
    try:
        dwi_data, dwi_affine = load_nifti(subject_row['dwi_file'])
        bvals, bvecs = read_bvals_bvecs(subject_row['bval_file'], subject_row['bvec_file'])
        gtab = gradient_table(bvals, bvecs)
        
        if config.VERBOSE:
            print(f"  [{sub_id}] DWI shape: {dwi_data.shape}, b-values: {np.unique(bvals)}")
        
        b0_idx = np.where(bvals < 50)[0][0]
        b0_data = dwi_data[..., b0_idx]
        
        masked_data, mask = median_otsu(
            dwi_data, 
            vol_idx=range(min(10, dwi_data.shape[-1])),
            median_radius=3,
            numpass=1
        )
        
        tenmodel = TensorModel(gtab)
        tenfit = tenmodel.fit(masked_data)
        
        fa = fractional_anisotropy(tenfit.evals)
        fa = np.clip(fa, 0, 1)
        fa[np.isnan(fa)] = 0
        
        stopping_criterion = ThresholdStoppingCriterion(fa, config.FA_THRESHOLD)
        
        from dipy.reconst.shm import CsaOdfModel
        from dipy.direction import peaks_from_model
        
        csa_model = CsaOdfModel(gtab, sh_order=4)
        csa_peaks = peaks_from_model(
            csa_model, 
            masked_data, 
            default_sphere,
            relative_peak_threshold=0.5,
            min_separation_angle=25,
            mask=mask
        )
        
        seed_mask = fa > 0.3
        seeds = tracking_utils.seeds_from_mask(
            seed_mask, 
            dwi_affine, 
            density=config.N_SEEDS_PER_VOXEL
        )
        
        if config.VERBOSE:
            print(f"  [{sub_id}] Seeds generated: {len(seeds)}")
        
        streamline_generator = LocalTracking(
            csa_peaks,
            stopping_criterion,
            seeds,
            dwi_affine,
            step_size=0.5,
            return_all=False
        )
        
        streamlines = Streamlines(streamline_generator)
        
        from dipy.tracking.streamline import length
        lengths = list(length(streamlines))
        long_streamlines = Streamlines([
            s for s, l in zip(streamlines, lengths) 
            if config.MIN_STREAMLINE_LENGTH < l < config.MAX_STREAMLINE_LENGTH
        ])
        
        if config.VERBOSE:
            print(f"  [{sub_id}] Streamlines: {len(streamlines)} → {len(long_streamlines)} (after filtering)")
        
        from dipy.align.imaffine import AffineMap
        from dipy.align.transforms import AffineTransform3D
        
        atlas_resampled = image.resample_to_img(
            atlas['img'],
            nib.Nifti1Image(fa, dwi_affine),
            interpolation='nearest'
        )
        atlas_data_subj = atlas_resampled.get_fdata().astype(int)
        
        n_regions = atlas['n_regions']
        sc_matrix = np.zeros((n_regions, n_regions), dtype=np.float32)
        
        for streamline in long_streamlines:
            start_point = streamline[0]
            end_point = streamline[-1]
            
            from dipy.tracking._utils import _to_voxel_coordinates
            start_vox = np.round(nib.affines.apply_affine(
                np.linalg.inv(dwi_affine), start_point
            )).astype(int)
            end_vox = np.round(nib.affines.apply_affine(
                np.linalg.inv(dwi_affine), end_point
            )).astype(int)
            
            shape = atlas_data_subj.shape
            if (0 <= start_vox[0] < shape[0] and 0 <= start_vox[1] < shape[1] and 
                0 <= start_vox[2] < shape[2] and
                0 <= end_vox[0] < shape[0] and 0 <= end_vox[1] < shape[1] and 
                0 <= end_vox[2] < shape[2]):
                
                region_start = atlas_data_subj[start_vox[0], start_vox[1], start_vox[2]]
                region_end = atlas_data_subj[end_vox[0], end_vox[1], end_vox[2]]
                
                if region_start > 0 and region_end > 0 and region_start != region_end:
                    i, j = int(region_start) - 1, int(region_end) - 1
                    if i < n_regions and j < n_regions:
                        sc_matrix[i, j] += 1
                        sc_matrix[j, i] += 1  # Symmetric
        
        
        sc_matrix_log = np.log1p(sc_matrix)
        
        if config.VERBOSE:
            print(f"  [{sub_id}] SC matrix: {np.sum(sc_matrix > 0)} connections, "
                  f"density: {np.sum(sc_matrix > 0) / (n_regions * (n_regions-1)):.3f}")
        
        return {
            'subject_id': sub_id,
            'sc_matrix': sc_matrix,
            'sc_matrix_log': sc_matrix_log,
            'fa_mean': np.mean(fa[mask > 0]),
            'n_streamlines': len(long_streamlines),
            'success': True
        }
        
    except Exception as e:
        print(f"  [{sub_id}] ERROR: {str(e)}")
        return {
            'subject_id': sub_id,
            'success': False,
            'error': str(e)
        }


def process_fmri_subject(subject_row, atlas, config=Config):
    sub_id = subject_row['subject_id']
    
    try:
        fmri_img = nib.load(subject_row['fmri_file'])
        
        if config.VERBOSE:
            print(f"  [{sub_id}] fMRI shape: {fmri_img.shape}")
        
        if fmri_img.shape[-1] > config.N_DUMMY_SCANS:
            fmri_img = image.index_img(fmri_img, slice(config.N_DUMMY_SCANS, None))
        
        n_timepoints = fmri_img.shape[-1]
        
        fmri_smooth = image.smooth_img(fmri_img, fwhm=6)
        
        fmri_filtered = image.clean_img(
            fmri_smooth,
            detrend=True,
            standardize=True,
            low_pass=config.HIGH_FREQ,
            high_pass=config.LOW_FREQ,
            t_r=config.TR
        )
        
        masker = NiftiLabelsMasker(
            labels_img=atlas['maps'],
            standardize=True,
            detrend=False,  # Already done
            memory='nilearn_cache',
            memory_level=1
        )
        
        time_series = masker.fit_transform(fmri_filtered)
        
        if config.VERBOSE:
            print(f"  [{sub_id}] Time series shape: {time_series.shape}")
        
        plv_matrix = compute_plv_matrix_gpu(time_series) if GPU_AVAILABLE else compute_plv_matrix(time_series)
        
        correlation_measure = ConnectivityMeasure(kind='correlation')
        corr_matrix = correlation_measure.fit_transform([time_series])[0]
        
        if config.VERBOSE:
            print(f"  [{sub_id}] PLV matrix computed, mean PLV: {np.mean(plv_matrix):.3f}")
        
        return {
            'subject_id': sub_id,
            'plv_matrix': plv_matrix,
            'corr_matrix': corr_matrix,
            'n_timepoints': n_timepoints,
            'time_series': time_series if config.SAVE_INDIVIDUAL else None,
            'success': True
        }
        
    except Exception as e:
        print(f"  [{sub_id}] ERROR: {str(e)}")
        return {
            'subject_id': sub_id,
            'success': False,
            'error': str(e)
        }


def compute_plv_matrix(time_series):
    from scipy.signal import hilbert
    
    n_regions, n_timepoints = time_series.shape[1], time_series.shape[0]
    
    analytic_signal = hilbert(time_series, axis=0)
    phases = np.angle(analytic_signal)
    
    plv_matrix = np.zeros((n_regions, n_regions), dtype=np.float32)
    
    for i in range(n_regions):
        for j in range(i+1, n_regions):
            phase_diff = phases[:, i] - phases[:, j]
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            plv_matrix[i, j] = plv
            plv_matrix[j, i] = plv
    
    np.fill_diagonal(plv_matrix, 1.0)
    
    return plv_matrix


def compute_plv_matrix_gpu(time_series):
    ts_gpu = cp.asarray(time_series)
    n_timepoints, n_regions = ts_gpu.shape
    
    fft_ts = cp.fft.fft(ts_gpu, axis=0)
    
    h = cp.zeros(n_timepoints)
    if n_timepoints % 2 == 0:
        h[0] = 1
        h[1:n_timepoints//2] = 2
        h[n_timepoints//2] = 1
    else:
        h[0] = 1
        h[1:(n_timepoints+1)//2] = 2
    
    analytic_signal = cp.fft.ifft(fft_ts * h[:, None], axis=0)
    phases = cp.angle(analytic_signal)
    
    
    plv_matrix = cp.zeros((n_regions, n_regions), dtype=cp.float32)
    
    chunk_size = 20  # Process 20 regions at a time
    for i_start in range(0, n_regions, chunk_size):
        i_end = min(i_start + chunk_size, n_regions)
        phases_i = phases[:, i_start:i_end, None]  # (T, chunk, 1)
        phases_j = phases[:, None, :]  # (T, 1, N)
        
        phase_diff = phases_i - phases_j  # (T, chunk, N)
        plv_chunk = cp.abs(cp.mean(cp.exp(1j * phase_diff), axis=0))  # (chunk, N)
        
        plv_matrix[i_start:i_end, :] = plv_chunk
    
    plv_matrix = (plv_matrix + plv_matrix.T) / 2
    cp.fill_diagonal(plv_matrix, 1.0)
    
    return cp.asnumpy(plv_matrix)


def compute_coupling_potential_metrics(sc_matrix, use_gpu=GPU_AVAILABLE):
    A = (sc_matrix > 0).astype(np.float32)
    n = A.shape[0]
    
    if use_gpu:
        A_gpu = cp.asarray(A)
        return _compute_cp_metrics_gpu(A_gpu, n)
    else:
        return _compute_cp_metrics_cpu(A, n)


def _compute_cp_metrics_cpu(A, n):
    
    metrics = {}
    
    degree = np.sum(A, axis=1)
    
    cn_matrix = A @ A
    np.fill_diagonal(cn_matrix, 0)
    metrics['common_neighbors'] = cn_matrix
    
    union_matrix = degree[:, None] + degree[None, :] - cn_matrix
    jaccard = np.divide(cn_matrix, union_matrix, 
                        out=np.zeros_like(cn_matrix), 
                        where=union_matrix > 0)
    np.fill_diagonal(jaccard, 0)
    metrics['jaccard'] = jaccard
    
    log_degree = np.log(degree + 1)  # +1 to avoid log(0)
    log_degree[log_degree < 1e-10] = 1e-10  # Avoid division by zero
    inv_log_degree = 1.0 / log_degree
    
    aa_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        neighbors_i = np.where(A[i] > 0)[0]
        for j in range(i+1, n):
            neighbors_j = np.where(A[j] > 0)[0]
            common = np.intersect1d(neighbors_i, neighbors_j)
            if len(common) > 0:
                aa_matrix[i, j] = np.sum(inv_log_degree[common])
                aa_matrix[j, i] = aa_matrix[i, j]
    metrics['adamic_adar'] = aa_matrix
    
    inv_degree = 1.0 / (degree + 1e-10)
    ra_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        neighbors_i = np.where(A[i] > 0)[0]
        for j in range(i+1, n):
            neighbors_j = np.where(A[j] > 0)[0]
            common = np.intersect1d(neighbors_i, neighbors_j)
            if len(common) > 0:
                ra_matrix[i, j] = np.sum(inv_degree[common])
                ra_matrix[j, i] = ra_matrix[i, j]
    metrics['resource_allocation'] = ra_matrix
    
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1e-10
    A_normalized = A / norms
    profile_sim = A_normalized @ A_normalized.T
    np.fill_diagonal(profile_sim, 0)
    metrics['profile_similarity'] = profile_sim
    
    eigenvalues = np.linalg.eigvalsh(A)
    lambda_max = np.max(eigenvalues)
    if lambda_max > 0:
        beta = 1.0 / lambda_max
    else:
        beta = 0.1
    
    comm_matrix = expm(beta * A)
    np.fill_diagonal(comm_matrix, 0)
    comm_matrix = comm_matrix * (1 - A)
    metrics['communicability'] = comm_matrix
    
    hub_threshold = np.percentile(degree, 90)
    hub_mask = degree >= hub_threshold
    A_hub = A[:, hub_mask]
    if A_hub.shape[1] > 0:
        hub_norms = np.linalg.norm(A_hub, axis=1, keepdims=True)
        hub_norms[hub_norms < 1e-10] = 1e-10
        A_hub_normalized = A_hub / hub_norms
        hub_affinity = A_hub_normalized @ A_hub_normalized.T
    else:
        hub_affinity = np.zeros((n, n))
    np.fill_diagonal(hub_affinity, 0)
    metrics['hub_affinity'] = hub_affinity
    
    composite = np.zeros((n, n), dtype=np.float32)
    for name, matrix in metrics.items():
        if np.max(matrix) > 0:
            normalized = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix) + 1e-10)
            composite += normalized
    composite /= len(metrics)
    metrics['composite'] = composite
    
    return metrics

def _compute_cp_metrics_gpu(A_gpu, n):
    
    metrics = {}
    
    degree = cp.sum(A_gpu, axis=1)
    
    cn_matrix = A_gpu @ A_gpu
    cp.fill_diagonal(cn_matrix, 0)
    metrics['common_neighbors'] = cp.asnumpy(cn_matrix)
    
    union_matrix = degree[:, None] + degree[None, :] - cn_matrix
    union_safe = cp.where(union_matrix > 0, union_matrix, cp.ones_like(union_matrix))
    jaccard = cn_matrix / union_safe
    jaccard = cp.where(union_matrix > 0, jaccard, cp.zeros_like(jaccard))
    cp.fill_diagonal(jaccard, 0)
    metrics['jaccard'] = cp.asnumpy(jaccard)
    
    norms = cp.linalg.norm(A_gpu, axis=1, keepdims=True)
    norms = cp.maximum(norms, 1e-10)
    A_normalized = A_gpu / norms
    profile_sim = A_normalized @ A_normalized.T
    cp.fill_diagonal(profile_sim, 0)
    metrics['profile_similarity'] = cp.asnumpy(profile_sim)
    
    A_cpu = cp.asnumpy(A_gpu)
    eigenvalues = np.linalg.eigvalsh(A_cpu)
    lambda_max = np.max(eigenvalues)
    beta = 1.0 / max(lambda_max, 0.1)
    comm_matrix = expm(beta * A_cpu)
    np.fill_diagonal(comm_matrix, 0)
    comm_matrix = comm_matrix * (1 - A_cpu)
    metrics['communicability'] = comm_matrix
    
    degree_cpu = cp.asnumpy(degree)
    log_degree = np.log(degree_cpu + 1)
    log_degree[log_degree < 1e-10] = 1e-10
    inv_log_degree = 1.0 / log_degree
    
    inv_degree = 1.0 / (degree_cpu + 1e-10)
    
    aa_matrix = np.zeros((n, n), dtype=np.float32)
    ra_matrix = np.zeros((n, n), dtype=np.float32)
    
    for i in range(n):
        neighbors_i = np.where(A_cpu[i] > 0)[0]
        for j in range(i+1, n):
            neighbors_j = np.where(A_cpu[j] > 0)[0]
            common = np.intersect1d(neighbors_i, neighbors_j)
            if len(common) > 0:
                aa_matrix[i, j] = np.sum(inv_log_degree[common])
                aa_matrix[j, i] = aa_matrix[i, j]
                ra_matrix[i, j] = np.sum(inv_degree[common])
                ra_matrix[j, i] = ra_matrix[i, j]
    
    metrics['adamic_adar'] = aa_matrix
    metrics['resource_allocation'] = ra_matrix
    
    hub_threshold = np.percentile(degree_cpu, 90)
    hub_mask = degree_cpu >= hub_threshold
    A_hub = A_cpu[:, hub_mask]
    if A_hub.shape[1] > 0:
        hub_norms = np.linalg.norm(A_hub, axis=1, keepdims=True)
        hub_norms[hub_norms < 1e-10] = 1e-10
        A_hub_normalized = A_hub / hub_norms
        hub_affinity = A_hub_normalized @ A_hub_normalized.T
    else:
        hub_affinity = np.zeros((n, n))
    np.fill_diagonal(hub_affinity, 0)
    metrics['hub_affinity'] = hub_affinity
    
    composite = np.zeros((n, n), dtype=np.float32)
    for name, matrix in metrics.items():
        if np.max(matrix) > 0:
            normalized = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix) + 1e-10)
            composite += normalized
    composite /= len(metrics)
    metrics['composite'] = composite
    
    return metrics


def run_pipeline(subjects_df, atlas, config=Config):
    
    results = {
        'subjects': [],
        'sc_matrices': [],
        'fc_matrices': [],
        'cp_metrics': [],
        'metadata': {
            'n_subjects': len(subjects_df),
            'atlas': config.ATLAS_NAME,
            'n_parcels': config.N_PARCELS,
            'start_time': datetime.now().isoformat()
        }
    }
    
    checkpoint_file = config.CHECKPOINT_DIR / "pipeline_checkpoint.pkl"
    start_idx = 0
    
    if checkpoint_file.exists():
        print("Found checkpoint, resuming...")
        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)
            results = checkpoint['results']
            start_idx = checkpoint['last_idx'] + 1
        print(f"  Resuming from subject {start_idx}")
    
    total_subjects = len(subjects_df)
    
    for idx in range(start_idx, total_subjects):
        subject_row = subjects_df.iloc[idx]
        sub_id = subject_row['subject_id']
        
        print(f"\n[{idx+1}/{total_subjects}] Processing {sub_id}...")
        start_time = time.time()
        
        print(f"  DTI processing...")
        dti_result = process_dti_subject(subject_row, atlas, config)
        
        if not dti_result['success']:
            print(f"  ✗ DTI failed, skipping subject")
            continue
        
        print(f"  fMRI processing...")
        fmri_result = process_fmri_subject(subject_row, atlas, config)
        
        if not fmri_result['success']:
            print(f"  ✗ fMRI failed, skipping subject")
            continue
        
        print(f"  Computing coupling potential metrics...")
        cp_metrics = compute_coupling_potential_metrics(dti_result['sc_matrix'])
        
        results['subjects'].append(sub_id)
        results['sc_matrices'].append(dti_result['sc_matrix'])
        results['fc_matrices'].append({
            'plv': fmri_result['plv_matrix'],
            'corr': fmri_result['corr_matrix']
        })
        results['cp_metrics'].append(cp_metrics)
        
        elapsed = time.time() - start_time
        print(f"  ✓ Completed in {elapsed:.1f}s")
        
        if config.SAVE_INDIVIDUAL:
            np.savez_compressed(
                config.OUTPUT_DIR / "sc_matrices" / f"{sub_id}_sc.npz",
                sc=dti_result['sc_matrix'],
                sc_log=dti_result['sc_matrix_log']
            )
            np.savez_compressed(
                config.OUTPUT_DIR / "fc_matrices" / f"{sub_id}_fc.npz",
                plv=fmri_result['plv_matrix'],
                corr=fmri_result['corr_matrix']
            )
        
        if (idx + 1) % 5 == 0:
            print(f"  Saving checkpoint...")
            with open(checkpoint_file, 'wb') as f:
                pickle.dump({'results': results, 'last_idx': idx}, f)
        
        gc.collect()
        if GPU_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()
    
    results['metadata']['end_time'] = datetime.now().isoformat()
    results['metadata']['n_successful'] = len(results['subjects'])
    
    with open(config.OUTPUT_DIR / "all_results.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n{'='*60}")
    print(f"Pipeline complete!")
    print(f"  Successful subjects: {len(results['subjects'])}/{total_subjects}")
    print(f"  Results saved to: {config.OUTPUT_DIR}")
    
    return results


def run_validation_analysis(results, config=Config):
    
    print("\n" + "="*60)
    print("VALIDATION ANALYSIS")
    print("="*60)
    
    n_subjects = len(results['subjects'])
    n_regions = config.N_PARCELS
    
    valid_indices = []
    for i, sub_id in enumerate(results['subjects']):
        sc = results['sc_matrices'][i]
        plv = results['fc_matrices'][i]['plv']
        
        if sc.shape[0] == n_regions and plv.shape[0] == n_regions:
            valid_indices.append(i)
        else:
            print(f"  Skipping {sub_id}: SC shape {sc.shape}, PLV shape {plv.shape}")
    
    print(f"\nValid subjects with {n_regions} regions: {len(valid_indices)}/{n_subjects}")
    
    if len(valid_indices) < 10:
        print("ERROR: Not enough valid subjects!")
        return None, None, None, None
    
    
    all_correlations = {metric: [] for metric in results['cp_metrics'][valid_indices[0]].keys()}
    all_correlations['corr_fc'] = []  # Also test correlation-based FC
    
    for i in valid_indices:
        sub_id = results['subjects'][i]
        sc = results['sc_matrices'][i]
        plv = results['fc_matrices'][i]['plv']
        corr = results['fc_matrices'][i]['corr']
        cp_metrics = results['cp_metrics'][i]
        
        A = (sc > 0).astype(float)
        
        non_connected = (A == 0)
        np.fill_diagonal(non_connected, False)  # Exclude diagonal
        
        triu_idx = np.triu_indices(n_regions, k=1)
        mask = non_connected[triu_idx]
        
        plv_values = plv[triu_idx][mask]
        corr_values = corr[triu_idx][mask]
        
        for metric_name, metric_matrix in cp_metrics.items():
            metric_values = metric_matrix[triu_idx][mask]
            
            if len(plv_values) > 10 and np.std(metric_values) > 1e-10:
                r, p = stats.pearsonr(metric_values, plv_values)
                all_correlations[metric_name].append({'r': r, 'p': p, 'n': len(plv_values)})
            
        composite_values = cp_metrics['composite'][triu_idx][mask]
        if len(corr_values) > 10:
            r, p = stats.pearsonr(composite_values, corr_values)
            all_correlations['corr_fc'].append({'r': r, 'p': p, 'n': len(corr_values)})
    
    print(f"\nWithin-subject correlations (CP ↔ PLV, n={len(valid_indices)} subjects):")
    print(f"{'Metric':<25} {'Mean r':>10} {'Std r':>10} {'Mean p':>12}")
    print("-" * 60)
    
    summary_stats = {}
    for metric_name, corr_list in all_correlations.items():
        if len(corr_list) > 0:
            r_values = [c['r'] for c in corr_list]
            p_values = [c['p'] for c in corr_list]
            mean_r = np.mean(r_values)
            std_r = np.std(r_values)
            mean_p = np.mean(p_values)
            
            summary_stats[metric_name] = {
                'mean_r': mean_r,
                'std_r': std_r,
                'mean_p': mean_p,
                'r_values': r_values
            }
            
            print(f"{metric_name:<25} {mean_r:>10.3f} {std_r:>10.3f} {mean_p:>12.2e}")
    
    print(f"\n\nGroup-level analysis:")
    
    valid_sc = [results['sc_matrices'][i] for i in valid_indices]
    valid_plv = [results['fc_matrices'][i]['plv'] for i in valid_indices]
    valid_corr = [results['fc_matrices'][i]['corr'] for i in valid_indices]
    
    mean_sc = np.mean(valid_sc, axis=0)
    mean_plv = np.mean(valid_plv, axis=0)
    mean_corr = np.mean(valid_corr, axis=0)
    
    mean_cp_metrics = compute_coupling_potential_metrics(mean_sc)
    
    A_mean = (mean_sc > np.percentile(mean_sc[mean_sc > 0], 10)).astype(float)  # Threshold at 10th percentile
    non_connected_mean = (A_mean == 0)
    np.fill_diagonal(non_connected_mean, False)
    
    triu_idx = np.triu_indices(n_regions, k=1)
    mask = non_connected_mean[triu_idx]
    
    print(f"\nGroup-average CP ↔ PLV correlations:")
    print(f"{'Metric':<25} {'r':>10} {'p':>12} {'R²':>10}")
    print("-" * 60)
    
    plv_values = mean_plv[triu_idx][mask]
    
    group_stats = {}
    for metric_name, metric_matrix in mean_cp_metrics.items():
        metric_values = metric_matrix[triu_idx][mask]
        
        if np.std(metric_values) > 1e-10:
            r, p = stats.pearsonr(metric_values, plv_values)
            r2 = r ** 2
            
            group_stats[metric_name] = {'r': r, 'p': p, 'r2': r2}
            print(f"{metric_name:<25} {r:>10.3f} {p:>12.2e} {r2:>10.3f}")
    
    validation_results = {
        'within_subject': summary_stats,
        'group_level': group_stats,
        'n_subjects': len(valid_indices),
        'n_subjects_excluded': n_subjects - len(valid_indices),
        'n_pairs_per_subject': np.sum(mask)
    }
    
    with open(config.OUTPUT_DIR / "validation_results.pkl", 'wb') as f:
        pickle.dump(validation_results, f)
    
    return validation_results, mean_sc, mean_plv, mean_cp_metrics


def create_validation_figures(validation_results, mean_sc, mean_plv, mean_cp_metrics, config=Config):
    
    fig_dir = config.OUTPUT_DIR / "figures"
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    im0 = axes[0].imshow(np.log1p(mean_sc), cmap='hot', aspect='equal')
    axes[0].set_title('Structural Connectivity\n(log scale)', fontsize=12)
    axes[0].set_xlabel('Region')
    axes[0].set_ylabel('Region')
    plt.colorbar(im0, ax=axes[0], shrink=0.8)
    
    im1 = axes[1].imshow(mean_plv, cmap='RdBu_r', vmin=0, vmax=0.5, aspect='equal')
    axes[1].set_title('Functional Connectivity\n(PLV)', fontsize=12)
    axes[1].set_xlabel('Region')
    axes[1].set_ylabel('Region')
    plt.colorbar(im1, ax=axes[1], shrink=0.8)
    
    im2 = axes[2].imshow(mean_cp_metrics['communicability'], cmap='viridis', aspect='equal')
    axes[2].set_title('Coupling Potential\n(Communicability)', fontsize=12)
    axes[2].set_xlabel('Region')
    axes[2].set_ylabel('Region')
    plt.colorbar(im2, ax=axes[2], shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(fig_dir / "Fig1_SC_FC_CP_matrices.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics_to_plot = ['common_neighbors', 'jaccard', 'adamic_adar', 
                       'profile_similarity', 'communicability', 'composite']
    
    data_to_plot = []
    labels = []
    
    for metric in metrics_to_plot:
        if metric in validation_results['within_subject']:
            r_values = validation_results['within_subject'][metric]['r_values']
            data_to_plot.append(r_values)
            labels.append(metric.replace('_', '\n'))
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(data_to_plot)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Pearson r (CP vs PLV)', fontsize=12)
    ax.set_title('Within-Subject Structure-Function Correlations\n(Structurally Distant Pairs)', fontsize=14)
    
    for i, metric in enumerate(metrics_to_plot):
        if metric in validation_results['within_subject']:
            mean_r = validation_results['within_subject'][metric]['mean_r']
            if mean_r > 0.3:
                ax.text(i+1, max(data_to_plot[i]) + 0.05, '***', ha='center', fontsize=12)
            elif mean_r > 0.2:
                ax.text(i+1, max(data_to_plot[i]) + 0.05, '**', ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(fig_dir / "Fig2_within_subject_correlations.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    n_regions = mean_plv.shape[0]
    A_mean = (mean_sc > np.percentile(mean_sc[mean_sc > 0], 10)).astype(float)
    non_connected_mean = (A_mean == 0)
    np.fill_diagonal(non_connected_mean, False)
    triu_idx = np.triu_indices(n_regions, k=1)
    mask = non_connected_mean[triu_idx]
    
    plv_values = mean_plv[triu_idx][mask]
    
    best_metric = max(validation_results['group_level'].items(), 
                      key=lambda x: x[1]['r'])[0]
    
    best_values = mean_cp_metrics[best_metric][triu_idx][mask]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    n_points = len(plv_values)
    if n_points > 5000:
        idx = np.random.choice(n_points, 5000, replace=False)
        plv_plot = plv_values[idx]
        cp_plot = best_values[idx]
    else:
        plv_plot = plv_values
        cp_plot = best_values
    
    ax.scatter(cp_plot, plv_plot, alpha=0.3, s=10, c='steelblue')
    
    slope, intercept, r, p, se = stats.linregress(cp_plot, plv_plot)
    x_line = np.linspace(np.min(cp_plot), np.max(cp_plot), 100)
    ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2, 
            label=f'r = {r:.3f}, R² = {r**2:.3f}')
    
    ax.set_xlabel(f'{best_metric.replace("_", " ").title()}', fontsize=12)
    ax.set_ylabel('Phase-Locking Value (PLV)', fontsize=12)
    ax.set_title(f'Group-Level Structure-Function Relationship\n(n = {n_points} non-connected pairs)', fontsize=14)
    ax.legend(loc='upper left', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(fig_dir / "Fig3_group_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    metrics = ['common_neighbors', 'jaccard', 'adamic_adar', 
               'profile_similarity', 'communicability', 'resource_allocation']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        
        metric_values = mean_cp_metrics[metric][triu_idx][mask]
        
        ax.scatter(metric_values, plv_values, alpha=0.2, s=5, c='steelblue')
        
        if np.std(metric_values) > 1e-10:
            r, p = stats.pearsonr(metric_values, plv_values)
            ax.set_title(f'{metric.replace("_", " ").title()}\nr = {r:.3f}', fontsize=11)
        else:
            ax.set_title(f'{metric.replace("_", " ").title()}\n(no variance)', fontsize=11)
        
        ax.set_xlabel('Coupling Potential')
        ax.set_ylabel('PLV')
    
    plt.suptitle('Validation of Analytical Predictions: All Metrics', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(fig_dir / "Fig4_all_metrics_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = list(validation_results['group_level'].keys())
    r_values = [validation_results['group_level'][m]['r'] for m in metrics]
    
    colors = ['green' if r > 0.3 else 'orange' if r > 0.1 else 'red' for r in r_values]
    
    bars = ax.bar(range(len(metrics)), r_values, color=colors, edgecolor='black')
    
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m.replace('_', '\n') for m in metrics], fontsize=10)
    ax.set_ylabel('Pearson r', fontsize=12)
    ax.set_title('Coupling Potential Predicts Functional Connectivity\n(DLBS Empirical Validation)', fontsize=14)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axhline(y=0.3, color='green', linestyle='--', alpha=0.5, label='Strong (r > 0.3)')
    
    for bar, r in zip(bars, r_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{r:.2f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(fig_dir / "Fig5_summary_barplot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Figures saved to {fig_dir}")
    
    return fig_dir


def main():
    print("=" * 60)
    print("COUPLING POTENTIAL VALIDATION PIPELINE")
    print("Dallas Lifespan Brain Study")
    print("=" * 60)
    print(f"GPU available: {GPU_AVAILABLE}")
    print(f"Subjects: {len(subjects_df)}")
    print(f"Atlas: {Config.ATLAS_NAME} ({Config.N_PARCELS} parcels)")
    results = run_pipeline(subjects_df, atlas, Config)
    if len(results["subjects"]) < 10:
        print(f"Only {len(results['subjects'])} subjects completed.")
        return
    validation_results, mean_sc, mean_plv, mean_cp_metrics = run_validation_analysis(results, Config)
    create_validation_figures(validation_results, mean_sc, mean_plv, mean_cp_metrics, Config)
    if validation_results:
        best = max(validation_results["group_level"].items(), key=lambda x: x[1]["r"])
        print(f"Best predictor: {best[0]}")
        print(f"Best correlation: r = {best[1]['r']:.3f}")

if __name__ == "__main__":
    main()
