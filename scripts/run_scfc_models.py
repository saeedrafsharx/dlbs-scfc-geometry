
import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import expm, pinv
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.transform import Rotation
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

RNG_SEED = 42
PREVALENCE = 0.6
TEST_FRACTION = 0.30
ALPHA_GRID = np.logspace(-2, 4, 13)

def parse_args():
    parser = argparse.ArgumentParser(description="Held-out SC-FC model comparison on DLBS.")
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("DLBS_DATA_DIR", "./results")))
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("DLBS_MODEL_OUTPUT", "./results/scfc_models")))
    parser.add_argument("--participants", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument("--n-spins", type=int, default=100)
    parser.add_argument("--cv-repeats", type=int, default=10)
    return parser.parse_args()

args = parse_args()
DATA_DIR = args.data_dir
OUT_DIR = args.output_dir
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"
DAT_DIR = OUT_DIR / "data"
for d in (FIG_DIR, TAB_DIR, DAT_DIR):
    d.mkdir(parents=True, exist_ok=True)
RNG_SEED = args.seed

def load_subject(sub_id, data_dir):
    sc_path = data_dir / "sc_matrices" / f"{sub_id}_sc.npz"
    fc_path = data_dir / "fc_matrices" / f"{sub_id}_fc.npz"
    if not (sc_path.exists() and fc_path.exists()): return None
    with np.load(sc_path) as d: SC = d["sc"].astype(np.float64)
    with np.load(fc_path) as d:
        PLV = d["plv"].astype(np.float64)
        CORR = d["corr"].astype(np.float64)
    for M in (SC, PLV, CORR): np.fill_diagonal(M, 0.0)
    SC = 0.5*(SC+SC.T); PLV = 0.5*(PLV+PLV.T); CORR = 0.5*(CORR+CORR.T)
    if SC.shape != PLV.shape: return None
    return SC, PLV, CORR

sc_files = sorted((DATA_DIR / "sc_matrices").glob("sub-*_sc.npz"))
all_subjects = [f.name.replace("_sc.npz", "") for f in sc_files]
valid = [s for s in all_subjects if load_subject(s, DATA_DIR) is not None]

TEST_FRACTION = 0.30
rng2 = np.random.default_rng(RNG_SEED)
shuffled = list(valid); rng2.shuffle(shuffled)
n_test = int(len(shuffled) * TEST_FRACTION)
test_subjects  = sorted(shuffled[:n_test])
train_subjects = sorted(shuffled[n_test:])
print(f"Train: {len(train_subjects)}, Test: {len(test_subjects)}")

def load_stack(subs):
    SC, PLV, CORR = [], [], []
    for s in subs:
        out = load_subject(s, DATA_DIR)
        if out is None: continue
        SC.append(out[0]); PLV.append(out[1]); CORR.append(out[2])
    return np.stack(SC), np.stack(PLV), np.stack(CORR)
SC_train, PLV_train, CORR_train = load_stack(train_subjects)
SC_test,  PLV_test,  CORR_test  = load_stack(test_subjects)
N = SC_train.shape[1]
print(f"Train SC: {SC_train.shape}, Test SC: {SC_test.shape}")


PREVALENCE = 0.6
A = ((SC_train > 0).mean(axis=0) >= PREVALENCE).astype(np.float64)
np.fill_diagonal(A, 0.0)

PLV_test_mean  = PLV_test.mean(axis=0); np.fill_diagonal(PLV_test_mean, 0.0)
print(f"Consensus SC: density={A.sum()/(N*(N-1)):.4f}, edges={int(A.sum()/2)}")

def upper_tri(N): return np.triu(np.ones((N, N), bool), k=1)
Abin = (A > 0).astype(np.int8)
Dmat = shortest_path(Abin, directed=False, unweighted=True)
mask = upper_tri(N) & ((Dmat >= 2) | np.isinf(Dmat))
n_distant = int(mask.sum())
iu = np.triu_indices(N, k=1)
mask_idx = np.where(mask[iu])[0]
print(f"Distant pairs: {n_distant}")


label_path = DATA_DIR / "atlas" / "schaefer_2018" / "Schaefer2018_100Parcels_7Networks_order.txt"
labels_yeo7 = []
hemis = []
with open(label_path) as f:
    for line in f:
        parts = line.split()
        roi = parts[1]  # e.g., 7Networks_LH_Vis_1
        labels_yeo7.append(roi.split("_")[2])
        hemis.append(roi.split("_")[1])  # LH or RH
labels_yeo7 = np.array(labels_yeo7)
hemis = np.array(hemis)
unique_nets, yeo_int = np.unique(labels_yeo7, return_inverse=True)
same_yeo = (yeo_int[:, None] == yeo_int[None, :])
print(f"Hemispheres: {dict(zip(*np.unique(hemis, return_counts=True)))}")
print(f"Yeo networks: {dict(zip(*np.unique(labels_yeo7, return_counts=True)))}")


GEOMETRIC_AVAILABLE = False
try:
    from nilearn import datasets
    from nilearn.plotting import find_parcellation_cut_coords
    print("Fetching Schaefer atlas...")
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7,
                                                resolution_mm=2)
    centroids = find_parcellation_cut_coords(atlas.maps)
    print(f"Centroids: {centroids.shape}")
    GEOMETRIC_AVAILABLE = True
except Exception as e:
    print(f"Could not fetch atlas: {e}")
    print("CRITICAL: this notebook needs centroids. Re-run with internet enabled.")
    raise SystemExit


def gaussian_kernel_eigendecomp(coords, sigma=None):
    D = squareform(pdist(coords, metric="euclidean"))
    if sigma is None:
        sigma = np.median(D[D > 0])
    W = np.exp(-D**2 / (2 * sigma**2))
    np.fill_diagonal(W, 0.0)
    deg = W.sum(axis=1)
    d12 = 1.0 / np.sqrt(deg)
    L = np.eye(coords.shape[0]) - (d12[:, None] * W * d12[None, :])
    L = 0.5 * (L + L.T)
    eigvals, eigvecs = np.linalg.eigh(L)
    return eigvals, eigvecs, sigma, W

eigvals_geom, eigvecs_geom, sigma_geom, W_geom = gaussian_kernel_eigendecomp(centroids)
eigvals_geom = eigvals_geom[1:]
eigvecs_geom = eigvecs_geom[:, 1:]
print(f"Geometric kernel: sigma = {sigma_geom:.1f} mm, "
      f"eigvals range [{eigvals_geom.min():.4f}, {eigvals_geom.max():.4f}]")


deg_sc = A.sum(1)
with np.errstate(divide="ignore"):
    d12 = np.where(deg_sc > 0, 1.0/np.sqrt(deg_sc), 0.0)
L_sc = np.eye(N) - (d12[:, None] * A * d12[None, :])
L_sc = 0.5 * (L_sc + L_sc.T)
eigvals_sc, eigvecs_sc = np.linalg.eigh(L_sc)
eigvals_sc = eigvals_sc[1:]
eigvecs_sc = eigvecs_sc[:, 1:]
print(f"SC eigenmodes: {eigvecs_sc.shape}, "
      f"eigvals range [{eigvals_sc.min():.4f}, {eigvals_sc.max():.4f}]")


def common_neighbors(A): M = A @ A; np.fill_diagonal(M, 0); return M
def jaccard(A):
    inter = A @ A; deg = A.sum(1)
    union = deg[:, None] + deg[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        J = np.where(union > 0, inter/union, 0.0)
    np.fill_diagonal(J, 0); return J
def adamic_adar(A):
    deg = A.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(deg > 1, 1.0/np.log(deg), 0.0)
    M = A @ np.diag(w) @ A; np.fill_diagonal(M, 0); return M
def resource_allocation(A):
    deg = A.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(deg > 0, 1.0/deg, 0.0)
    M = A @ np.diag(w) @ A; np.fill_diagonal(M, 0); return M
def profile_similarity(A):
    n = np.linalg.norm(A, axis=1); n[n==0] = 1.0
    P = (A @ A.T) / np.outer(n, n); np.fill_diagonal(P, 0); return P
def communicability(A, beta=None):
    rho = float(np.max(np.abs(np.linalg.eigvalsh(A))))
    if beta is None: beta = 1.0/rho if rho > 0 else 1.0
    C = expm(beta * A); np.fill_diagonal(C, 0); return C
def hub_affinity(A, hub_quantile=0.9):
    deg = A.sum(1); is_hub = deg >= np.quantile(deg, hub_quantile)
    H = A[:, is_hub]; n = np.linalg.norm(H, axis=1); n[n==0] = 1.0
    S = (H @ H.T) / np.outer(n, n); np.fill_diagonal(S, 0); return S
def effective_resistance(A, normalized=False):
    deg = A.sum(1)
    if normalized:
        with np.errstate(divide="ignore"):
            d12 = np.where(deg > 0, 1.0/np.sqrt(deg), 0.0)
        L = np.eye(A.shape[0]) - (d12[:, None] * A * d12[None, :])
    else:
        L = np.diag(deg) - A
    Lp = pinv(L); diag = np.diag(Lp)
    R = diag[:, None] + diag[None, :] - 2*Lp
    np.fill_diagonal(R, 0); return R

CP_FEATURES = {
    "common_neighbors":     common_neighbors(A),
    "jaccard":              jaccard(A),
    "adamic_adar":          adamic_adar(A),
    "resource_allocation":  resource_allocation(A),
    "profile_similarity":   profile_similarity(A),
    "communicability":      communicability(A),
    "hub_affinity":         hub_affinity(A),
    "neg_R_combinatorial": -effective_resistance(A, normalized=False),
    "neg_R_symmetric":     -effective_resistance(A, normalized=True),
}

def cp_features_for_pairs(mm):
    return np.column_stack([P[mm] for P in CP_FEATURES.values()])

def spectral_features_for_pairs(eigvecs, K, mm):
    iu = np.triu_indices(eigvecs.shape[0], k=1)
    mm_idx = np.where(mm[iu])[0]
    feats = []
    for k in range(K):
        outer = np.outer(eigvecs[:, k], eigvecs[:, k])
        feats.append(outer[iu][mm_idx])
    return np.column_stack(feats)


def stack_targets(FC_stack, mm):
    return np.concatenate([FC_stack[s][mm] for s in range(FC_stack.shape[0])])

def tile_features(X_pairs, n_subjects):
    return np.tile(X_pairs, (n_subjects, 1))

ALPHA_GRID = np.logspace(-2, 4, 13)

def fit_predict(X_pairs, FC_train, FC_test, mm=None):
    if mm is None: mm = mask
    n_train = FC_train.shape[0]
    y_train_flat = stack_targets(FC_train, mm)
    X_train_tiled = tile_features(X_pairs, n_train)
    sc = StandardScaler().fit(X_train_tiled)
    cv = RidgeCV(alphas=ALPHA_GRID, cv=5).fit(sc.transform(X_train_tiled), y_train_flat)
    pair_pred = cv.predict(sc.transform(X_pairs))
    actual_per = [FC_test[s][mm] for s in range(FC_test.shape[0])]
    actual_flat = np.concatenate(actual_per)
    pred_flat = np.tile(pair_pred, FC_test.shape[0])
    r_stacked = stats.pearsonr(actual_flat, pred_flat)[0]
    actual_mean = FC_test.mean(axis=0)[mm]
    r_group = stats.pearsonr(actual_mean, pair_pred)[0]
    ss_res = np.sum((actual_flat - pred_flat)**2)
    ss_tot = np.sum((actual_flat - actual_flat.mean())**2)
    R2 = 1 - ss_res/ss_tot
    return {"alpha": float(cv.alpha_), "n_features": int(X_pairs.shape[1]),
            "pair_pred": pair_pred,
            "test_r_stacked": r_stacked, "test_r_group": r_group, "test_R2": R2}


def K_sensitivity(eigvecs, FC_stack, K_grid, mm, label):
    rows = []
    n_subj = FC_stack.shape[0]
    kf = KFold(n_splits=5, shuffle=True, random_state=RNG_SEED)
    for K in K_grid:
        X_pairs = spectral_features_for_pairs(eigvecs, K, mm)
        rs, r2s = [], []
        for tr, te in kf.split(np.arange(n_subj)):
            n_tr, n_te = len(tr), len(te)
            y_tr = np.concatenate([FC_stack[s][mm] for s in tr])
            y_te = np.concatenate([FC_stack[s][mm] for s in te])
            X_tr = tile_features(X_pairs, n_tr)
            X_te = tile_features(X_pairs, n_te)
            sc = StandardScaler().fit(X_tr)
            r = RidgeCV(alphas=ALPHA_GRID, cv=3).fit(sc.transform(X_tr), y_tr)
            yp = r.predict(sc.transform(X_te))
            rs.append(stats.pearsonr(y_te, yp)[0])
            ss_res = np.sum((y_te - yp)**2); ss_tot = np.sum((y_te - y_te.mean())**2)
            r2s.append(1 - ss_res/ss_tot)
        rows.append({"model": label, "K": K, "inner_r_mean": np.mean(rs),
                     "inner_r_sd": np.std(rs), "inner_R2_mean": np.mean(r2s)})
        print(f"  {label:14s} K={K:>3d}: inner-CV r = {np.mean(rs):.4f}  R² = {np.mean(r2s):.4f}")
    return pd.DataFrame(rows)

K_GRID = [10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
print("=== K sensitivity: SC spectral ===")
df_K_sc = K_sensitivity(eigvecs_sc, PLV_train, K_GRID, mask, "SC_spectral")
print("\n=== K sensitivity: Geometric ===")
df_K_geo = K_sensitivity(eigvecs_geom, PLV_train, K_GRID, mask, "Geometric")

df_K_all = pd.concat([df_K_sc, df_K_geo], ignore_index=True)
df_K_all.to_csv(TAB_DIR / "K_sensitivity.csv", index=False)

K_sc_best = int(df_K_sc.loc[df_K_sc["inner_r_mean"].idxmax(), "K"])
K_geo_best = int(df_K_geo.loc[df_K_geo["inner_r_mean"].idxmax(), "K"])
print(f"\nSelected K_sc = {K_sc_best},  K_geo = {K_geo_best}")


fig, ax = plt.subplots(figsize=(8, 5))
for name, color in [("SC_spectral", "#264653"), ("Geometric", "#e76f51")]:
    s = df_K_all[df_K_all["model"] == name]
    ax.errorbar(s["K"], s["inner_r_mean"], yerr=s["inner_r_sd"],
                marker="o", label=name, color=color, capsize=3)
ax.set_xlabel("Number of eigenmodes (K)")
ax.set_ylabel("Inner-CV Pearson r (5-fold)")
ax.set_title(f"K sensitivity sweep on training cohort\n"
             f"Selected: K_SC = {K_sc_best}, K_geo = {K_geo_best}")
ax.axvline(K_sc_best, color="#264653", linestyle="--", alpha=0.5)
ax.axvline(K_geo_best, color="#e76f51", linestyle="--", alpha=0.5)
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_K_sensitivity.png")
fig.savefig(FIG_DIR / "fig_K_sensitivity.pdf")



X_cp = cp_features_for_pairs(mask)
X_sc = spectral_features_for_pairs(eigvecs_sc, K_sc_best, mask)
X_geo = spectral_features_for_pairs(eigvecs_geom, K_geo_best, mask)

print("Fitting 7 models on TRAIN, evaluating on TEST...")
fits = {}
fits["CP_baseline"]      = fit_predict(X_cp,                        PLV_train, PLV_test)
fits["SC_spectral"]      = fit_predict(X_sc,                        PLV_train, PLV_test)
fits["Geometric"]        = fit_predict(X_geo,                       PLV_train, PLV_test)
fits["Hybrid"]           = fit_predict(np.column_stack([X_cp, X_sc, X_geo]), PLV_train, PLV_test)
fits["Hybrid_minus_CP"]  = fit_predict(np.column_stack([X_sc, X_geo]),       PLV_train, PLV_test)
fits["Hybrid_minus_SC"]  = fit_predict(np.column_stack([X_cp, X_geo]),       PLV_train, PLV_test)
fits["Hybrid_minus_Geo"] = fit_predict(np.column_stack([X_cp, X_sc]),        PLV_train, PLV_test)

df_fits = pd.DataFrame([{"model": k, **{kk: vv for kk, vv in v.items() if kk != "pair_pred"}}
                         for k, v in fits.items()])
print(df_fits.round(4).to_string(index=False))
df_fits.to_csv(TAB_DIR / "all_models_test.csv", index=False)


hybrid_r = fits["Hybrid"]["test_r_stacked"]
hybrid_r2 = fits["Hybrid"]["test_R2"]
ablation_records = []
for component, ablated_model in [
    ("CP_predictors", "Hybrid_minus_CP"),
    ("SC_spectral",   "Hybrid_minus_SC"),
    ("Geometric",     "Hybrid_minus_Geo"),
]:
    drop_r  = hybrid_r  - fits[ablated_model]["test_r_stacked"]
    drop_r2 = hybrid_r2 - fits[ablated_model]["test_R2"]
    ablation_records.append({
        "component_removed": component,
        "ablated_model": ablated_model,
        "ablated_r": fits[ablated_model]["test_r_stacked"],
        "ablated_R2": fits[ablated_model]["test_R2"],
        "drop_r":  drop_r,
        "drop_R2": drop_r2,
    })
df_ablation = pd.DataFrame(ablation_records)
df_ablation["pct_R2_explained_uniquely"] = 100 * df_ablation["drop_R2"] / hybrid_r2
print(f"\nFull Hybrid: r = {hybrid_r:.4f}, R² = {hybrid_r2:.4f}")
print(f"\n=== ABLATION: how much does removing each component drop performance? ===")
print(df_ablation.round(4).to_string(index=False))
df_ablation.to_csv(TAB_DIR / "ablation_analysis.csv", index=False)


fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
labels = ["Full\nHybrid", "- CP", "- SC_spectral", "- Geometric"]
rs = [hybrid_r,
      fits["Hybrid_minus_CP"]["test_r_stacked"],
      fits["Hybrid_minus_SC"]["test_r_stacked"],
      fits["Hybrid_minus_Geo"]["test_r_stacked"]]
r2s = [hybrid_r2,
       fits["Hybrid_minus_CP"]["test_R2"],
       fits["Hybrid_minus_SC"]["test_R2"],
       fits["Hybrid_minus_Geo"]["test_R2"]]
xpos = np.arange(len(labels))
ax.bar(xpos - 0.2, rs, width=0.4, label="Test r", color="#2a9d8f")
ax2 = ax.twinx()
ax2.bar(xpos + 0.2, r2s, width=0.4, label="Test R²", color="#e76f51")
ax.set_xticks(xpos); ax.set_xticklabels(labels)
ax.set_ylabel("Test Pearson r", color="#2a9d8f")
ax2.set_ylabel("Test R²", color="#e76f51")
ax.set_title("A. Hybrid model with each component removed")
ax.axhline(0, color="gray", lw=0.5); ax.grid(axis="y", alpha=0.3)

ax = axes[1]
xpos = np.arange(len(df_ablation))
ax.bar(xpos, df_ablation["drop_R2"], color="#264653")
ax.set_xticks(xpos)
ax.set_xticklabels(df_ablation["component_removed"], rotation=15, ha="right")
ax.set_ylabel("Drop in test R² when removed")
ax.set_title("B. Unique contribution of each component\n"
             f"(full Hybrid R² = {hybrid_r2:.3f})")
for x, v, pct in zip(xpos, df_ablation["drop_R2"],
                      df_ablation["pct_R2_explained_uniquely"]):
    ax.annotate(f"{pct:.0f}%", xy=(x, v), xytext=(0, 5),
                textcoords="offset points", ha="center")
ax.axhline(0, color="gray", lw=0.5); ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(FIG_DIR / "fig_ablation.png")
fig.savefig(FIG_DIR / "fig_ablation.pdf")



def spin_centroids_hemispheric(coords, hemis, seed):
    rng = np.random.default_rng(seed)
    new_coords = coords.copy()
    for h in ("LH", "RH"):
        idx = np.where(hemis == h)[0]
        if len(idx) < 2: continue
        c = coords[idx]
        center = c.mean(axis=0)
        c_centered = c - center
        R_mat = Rotation.random(random_state=rng.integers(0, 1_000_000)).as_matrix()
        c_rot = c_centered @ R_mat.T
        new_coords[idx] = c_rot + center
    return new_coords

def fit_geom_fixed_alpha(coords, K, alpha, FC_train, FC_test, mm=None):
    if mm is None: mm = mask
    _, evec, _, _ = gaussian_kernel_eigendecomp(coords)
    evec = evec[:, 1:]
    X_pairs = spectral_features_for_pairs(evec, K, mm)

    n_train = FC_train.shape[0]
    y_train_flat = stack_targets(FC_train, mm)
    X_train_tiled = tile_features(X_pairs, n_train)
    sc = StandardScaler().fit(X_train_tiled)
    rid = Ridge(alpha=alpha).fit(sc.transform(X_train_tiled), y_train_flat)
    pair_pred = rid.predict(sc.transform(X_pairs))

    actual_per = [FC_test[s][mm] for s in range(FC_test.shape[0])]
    actual_flat = np.concatenate(actual_per)
    pred_flat = np.tile(pair_pred, FC_test.shape[0])
    r_stacked = stats.pearsonr(actual_flat, pred_flat)[0]
    ss_res = np.sum((actual_flat - pred_flat)**2)
    ss_tot = np.sum((actual_flat - actual_flat.mean())**2)
    R2 = 1 - ss_res/ss_tot
    return {"test_r_stacked": r_stacked, "test_R2": R2}

real_geom_r  = fits["Geometric"]["test_r_stacked"]
real_geom_r2 = fits["Geometric"]["test_R2"]
alpha_geom   = fits["Geometric"]["alpha"]
print(f"Real Geometric: r = {real_geom_r:.4f},  R² = {real_geom_r2:.4f}, alpha = {alpha_geom:.4f}")

N_SPINS = args.n_spins
print(f"\nRunning {N_SPINS} hemispheric spin nulls (fixed alpha for speed)...")
spin_records = []
for i in range(N_SPINS):
    coords_spun = spin_centroids_hemispheric(centroids, hemis, seed=RNG_SEED + i)
    res = fit_geom_fixed_alpha(coords_spun, K_geo_best, alpha_geom, PLV_train, PLV_test)
    spin_records.append({"iter": i, "null": "hemispheric_spin",
                         "test_r": res["test_r_stacked"],
                         "test_R2": res["test_R2"]})
    if (i+1) % 25 == 0:
        print(f"  spin {i+1}/{N_SPINS} done, last r = {res['test_r_stacked']:.3f}")

print(f"\nRunning {N_SPINS} parcel-permutation nulls...")
for i in range(N_SPINS):
    rng = np.random.default_rng(RNG_SEED + 10_000 + i)
    perm = rng.permutation(N)
    coords_perm = centroids[perm]
    res = fit_geom_fixed_alpha(coords_perm, K_geo_best, alpha_geom, PLV_train, PLV_test)
    spin_records.append({"iter": i, "null": "parcel_permutation",
                         "test_r": res["test_r_stacked"],
                         "test_R2": res["test_R2"]})
    if (i+1) % 25 == 0:
        print(f"  perm {i+1}/{N_SPINS} done, last r = {res['test_r_stacked']:.3f}")

df_spin = pd.DataFrame(spin_records)
df_spin.to_csv(TAB_DIR / "spin_null_distributions.csv", index=False)


spin_summary = []
for null_name in df_spin["null"].unique():
    s = df_spin[df_spin["null"] == null_name]
    null_r = s["test_r"].values
    null_R2 = s["test_R2"].values
    p_r  = (np.sum(np.abs(null_r)  >= abs(real_geom_r)) + 1)  / (len(null_r) + 1)
    p_R2 = (np.sum(null_R2 >= real_geom_r2) + 1) / (len(null_R2) + 1)
    z_r  = (real_geom_r  - null_r.mean())  / (null_r.std()  + 1e-10)
    z_R2 = (real_geom_r2 - null_R2.mean()) / (null_R2.std() + 1e-10)
    spin_summary.append({
        "null": null_name,
        "n_iter": len(null_r),
        "real_r": real_geom_r,
        "null_r_mean": null_r.mean(),
        "null_r_sd":   null_r.std(),
        "z_r":   z_r,
        "p_r_one_sided":  p_r,
        "real_R2": real_geom_r2,
        "null_R2_mean": null_R2.mean(),
        "z_R2":  z_R2,
        "p_R2_one_sided": p_R2,
    })
df_spin_summary = pd.DataFrame(spin_summary)
print("=== SPIN-TEST NULL SUMMARY ===")
print(df_spin_summary.round(4).to_string(index=False))
df_spin_summary.to_csv(TAB_DIR / "spin_null_summary.csv", index=False)


fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, metric, real_val in [
    (axes[0], "test_r",  real_geom_r),
    (axes[1], "test_R2", real_geom_r2),
]:
    for null_name, color in [
        ("hemispheric_spin",   "#264653"),
        ("parcel_permutation", "#e76f51"),
    ]:
        s = df_spin[df_spin["null"] == null_name][metric]
        ax.hist(s, bins=20, alpha=0.6, color=color,
                label=f"{null_name} null (mean={s.mean():.3f})")
    ax.axvline(real_val, color="black", lw=2.5,
               label=f"Real Geometric ({real_val:.3f})")
    ax.set_xlabel(metric); ax.set_ylabel("Count")
    ax.set_title(f"Spin null: {metric}")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

fig.suptitle(f"Spin-test null distributions for the Geometric model "
             f"({N_SPINS} iterations each)", y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_spin_null.png")
fig.savefig(FIG_DIR / "fig_spin_null.pdf")



fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.4)

main_models = ["CP_baseline", "SC_spectral", "Geometric", "Hybrid"]
df_main = df_fits[df_fits["model"].isin(main_models)].set_index("model").loc[main_models].reset_index()

ax = fig.add_subplot(gs[0, 0])
xpos = np.arange(len(df_main))
ax.bar(xpos - 0.2, df_main["test_r_stacked"], width=0.4,
       label="Test r (stacked)", color="#2a9d8f")
ax.bar(xpos + 0.2, df_main["test_r_group"],   width=0.4,
       label="Test r (group mean)", color="#e76f51")
ax.set_xticks(xpos); ax.set_xticklabels(main_models, rotation=15, ha="right")
ax.set_ylabel("Held-out Pearson r")
ax.set_title("A. Held-out test performance")
ax.legend(fontsize=8); ax.axhline(0, color="gray", lw=0.5)
ax.grid(axis="y", alpha=0.3)

ax = fig.add_subplot(gs[0, 1])
for name, color in [("SC_spectral", "#264653"), ("Geometric", "#e76f51")]:
    s = df_K_all[df_K_all["model"] == name]
    ax.errorbar(s["K"], s["inner_r_mean"], yerr=s["inner_r_sd"],
                marker="o", label=name, color=color, capsize=3)
ax.set_xlabel("K (number of eigenmodes)")
ax.set_ylabel("Inner-CV Pearson r")
ax.set_title("B. K sensitivity (training cohort)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[0, 2])
xpos = np.arange(len(df_ablation))
ax.bar(xpos, df_ablation["drop_R2"], color="#264653")
for x, v, pct in zip(xpos, df_ablation["drop_R2"],
                      df_ablation["pct_R2_explained_uniquely"]):
    ax.annotate(f"{pct:.0f}%", xy=(x, v), xytext=(0, 5),
                textcoords="offset points", ha="center", fontsize=9)
ax.set_xticks(xpos)
ax.set_xticklabels(df_ablation["component_removed"], rotation=15, ha="right")
ax.set_ylabel("Drop in test R² when removed")
ax.set_title(f"C. Unique R² contribution\n(full Hybrid R² = {hybrid_r2:.3f})")
ax.axhline(0, color="gray", lw=0.5); ax.grid(axis="y", alpha=0.3)

ax = fig.add_subplot(gs[1, 0])
for null_name, color in [
    ("hemispheric_spin",   "#264653"),
    ("parcel_permutation", "#e76f51"),
]:
    s = df_spin[df_spin["null"] == null_name]["test_r"]
    ax.hist(s, bins=20, alpha=0.6, color=color,
            label=f"{null_name}\n(p={df_spin_summary[df_spin_summary['null']==null_name]['p_r_one_sided'].iloc[0]:.4f})")
ax.axvline(real_geom_r, color="black", lw=2.5,
           label=f"Real ({real_geom_r:.3f})")
ax.set_xlabel("Test r (Geometric model)")
ax.set_ylabel("Count")
ax.set_title("D. Spin-test null for Geometric")
ax.legend(loc="upper left", fontsize=7); ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[1, 1])
def dist_strat(model_name):
    pred = fits[model_name]["pair_pred"]
    pred_full = np.zeros((N, N))
    pred_full[iu[0][mask_idx], iu[1][mask_idx]] = pred
    pred_full = pred_full + pred_full.T
    actual_mean = PLV_test.mean(axis=0)
    rs = []
    for d in (2, 3, 4, 5):
        dmask = upper_tri(N) & (Dmat == d)
        if dmask.sum() < 30:
            rs.append(np.nan); continue
        rs.append(stats.pearsonr(pred_full[dmask], actual_mean[dmask])[0])
    return rs

colors = {"CP_baseline": "#2a9d8f", "SC_spectral": "#e76f51",
          "Geometric": "#7b68ee", "Hybrid": "#000000"}
for m in main_models:
    rs = dist_strat(m)
    ax.plot([2,3,4,5], rs, marker="o", label=m, color=colors[m])
ax.set_xlabel("SC shortest-path distance")
ax.set_ylabel("Held-out test r (group mean)")
ax.set_title("E. Distance-stratified test performance")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax.axhline(0, color="gray", lw=0.5)

ax = fig.add_subplot(gs[1, 2])
hybrid_pred = fits["Hybrid"]["pair_pred"]
sub_rs = []
for s in range(PLV_test.shape[0]):
    actual = PLV_test[s][mask]
    if actual.std() > 0:
        sub_rs.append(stats.pearsonr(hybrid_pred, actual)[0])
sub_rs = np.array(sub_rs)
ax.hist(sub_rs, bins=20, color="#264653", alpha=0.8)
ax.axvline(sub_rs.mean(), color="red", lw=2, label=f"mean = {sub_rs.mean():.3f}")
ax.set_xlabel("Per-subject test r (Hybrid)")
ax.set_ylabel("Count")
ax.set_title(f"F. Per-subject distribution (n_test = {len(sub_rs)})")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.suptitle(f"Final SC-FC model comparison on DLBS "
             f"(train n = {PLV_train.shape[0]}, test n = {PLV_test.shape[0]})",
             y=1.0, fontsize=14, fontweight="bold")
fig.savefig(FIG_DIR / "fig_master_paper.png")
fig.savefig(FIG_DIR / "fig_master_paper.pdf")



def to_native(v):
    if isinstance(v, (np.floating, np.integer)): return v.item()
    if isinstance(v, np.ndarray): return v.tolist()
    return v

final_summary = {
    "split": {"n_train": int(PLV_train.shape[0]),
              "n_test":  int(PLV_test.shape[0]),
              "test_fraction": TEST_FRACTION, "seed": RNG_SEED},
    "data": {"n_nodes": int(N), "consensus_prevalence": PREVALENCE,
             "n_distant_pairs": int(n_distant)},
    "hyperparameters": {
        "K_sc_best":   K_sc_best,
        "K_geo_best":  K_geo_best,
        "alphas": {k: v["alpha"] for k, v in fits.items()},
    },
    "main_results": {
        m: {
            "test_r_stacked": fits[m]["test_r_stacked"],
            "test_r_group":   fits[m]["test_r_group"],
            "test_R2":        fits[m]["test_R2"],
        } for m in main_models
    },
    "ablation": df_ablation.set_index("component_removed").to_dict(orient="index"),
    "spin_null_summary": df_spin_summary.set_index("null").to_dict(orient="index"),
    "interpretation": {
        "hybrid_test_r":        hybrid_r,
        "hybrid_test_R2":       hybrid_r2,
        "winning_model":        df_fits.sort_values("test_r_stacked", ascending=False).iloc[0]["model"],
        "geometry_significant_against_spin_null": bool(
            df_spin_summary[df_spin_summary["null"]=="hemispheric_spin"]["p_r_one_sided"].iloc[0] < 0.05),
        "geometry_significant_against_perm_null": bool(
            df_spin_summary[df_spin_summary["null"]=="parcel_permutation"]["p_r_one_sided"].iloc[0] < 0.05),
    },
}
with open(OUT_DIR / "final_summary.json", "w") as f:
    json.dump(final_summary, f, indent=2, default=to_native)

np.savez_compressed(DAT_DIR / "final_artifacts.npz",
    A_consensus=A,
    PLV_test_mean=PLV_test_mean,
    centroids=centroids,
    eigvecs_sc=eigvecs_sc, eigvecs_geom=eigvecs_geom,
    labels_yeo=yeo_int, hemis=hemis,
    **{f"pred_{k}": v["pair_pred"] for k, v in fits.items()})

print(f"\nAll outputs saved to {OUT_DIR.resolve()}")
print(f"Hybrid: r = {hybrid_r:.4f}, R² = {hybrid_r2:.4f}")
print(f"Geometry vs spin null: p = "
      f"{df_spin_summary[df_spin_summary['null']=='hemispheric_spin']['p_r_one_sided'].iloc[0]:.4f}")
print(f"Geometry vs perm null: p = "
      f"{df_spin_summary[df_spin_summary['null']=='parcel_permutation']['p_r_one_sided'].iloc[0]:.4f}")


import pandas as pd
from scipy import stats as sstats


PARTICIPANTS_TSV = args.participants
if PARTICIPANTS_TSV:
    demo = pd.read_csv(PARTICIPANTS_TSV, sep="\t")
    id_col = next((c for c in demo.columns if c.lower() in ("participant_id", "sub_id", "subject", "subject_id", "id")), None)
    age_col = next((c for c in demo.columns if "age" in c.lower()), None)
    if id_col and age_col:
        demo_idx = demo.set_index(id_col)
        demo_idx.index = [str(i) if str(i).startswith("sub-") else f"sub-{i}" for i in demo_idx.index]
        train_ids = [s for s in train_subjects if s in demo_idx.index]
        test_ids = [s for s in test_subjects if s in demo_idx.index]
        train_age = demo_idx.loc[train_ids, age_col].astype(float)
        test_age = demo_idx.loc[test_ids, age_col].astype(float)
        t, p = stats.ttest_ind(train_age, test_age)
        u, p_mw = stats.mannwhitneyu(train_age, test_age)
        pd.DataFrame([{
            "id_col": id_col, "age_col": age_col,
            "train_mean_age": train_age.mean(), "train_sd_age": train_age.std(), "n_train": len(train_age),
            "test_mean_age": test_age.mean(), "test_sd_age": test_age.std(), "n_test": len(test_age),
            "ttest_p": p, "mannwhitney_p": p_mw,
        }]).to_csv(TAB_DIR / "age_balance.csv", index=False)

N_REPEATS = args.cv_repeats          # bump to 20+ once you've timed one pass
TEST_FRACTION_CV = 0.30

def build_split(seed):
    rng = np.random.default_rng(seed)
    shuffled = list(valid); rng.shuffle(shuffled)
    n_test = int(len(shuffled) * TEST_FRACTION_CV)
    return sorted(shuffled[n_test:]), sorted(shuffled[:n_test])

def run_one_split(seed):
    train_s, test_s = build_split(seed)
    SC_tr, PLV_tr, _ = load_stack(train_s)
    SC_te, PLV_te, _ = load_stack(test_s)

    A_ = ((SC_tr > 0).mean(axis=0) >= PREVALENCE).astype(np.float64)
    np.fill_diagonal(A_, 0.0)
    Abin_ = (A_ > 0).astype(np.int8)
    Dmat_ = shortest_path(Abin_, directed=False, unweighted=True)
    mask_ = upper_tri(N) & ((Dmat_ >= 2) | np.isinf(Dmat_))

    deg_ = A_.sum(1)
    with np.errstate(divide="ignore"):
        d12_ = np.where(deg_ > 0, 1.0/np.sqrt(deg_), 0.0)
    L_ = np.eye(N) - (d12_[:, None] * A_ * d12_[None, :])
    L_ = 0.5*(L_ + L_.T)
    _, eigvecs_ = np.linalg.eigh(L_)
    eigvecs_ = eigvecs_[:, 1:]

    cp_feats_ = {
        "common_neighbors":    common_neighbors(A_),
        "jaccard":             jaccard(A_),
        "adamic_adar":         adamic_adar(A_),
        "resource_allocation": resource_allocation(A_),
        "profile_similarity":  profile_similarity(A_),
        "communicability":     communicability(A_),
        "hub_affinity":        hub_affinity(A_),
        "neg_R_combinatorial": -effective_resistance(A_, normalized=False),
        "neg_R_symmetric":     -effective_resistance(A_, normalized=True),
    }
    X_cp_  = np.column_stack([P[mask_] for P in cp_feats_.values()])
    X_sc_  = spectral_features_for_pairs(eigvecs_, K_sc_best, mask_)
    X_geo_ = spectral_features_for_pairs(eigvecs_geom, K_geo_best, mask_)

    out = {}
    out["CP_baseline"] = fit_predict(X_cp_, PLV_tr, PLV_te, mm=mask_)
    out["SC_spectral"] = fit_predict(X_sc_, PLV_tr, PLV_te, mm=mask_)
    out["Geometric"]   = fit_predict(X_geo_, PLV_tr, PLV_te, mm=mask_)
    out["Hybrid"]      = fit_predict(np.column_stack([X_cp_, X_sc_, X_geo_]), PLV_tr, PLV_te, mm=mask_)
    return {k: {"test_r_stacked": v["test_r_stacked"],
                "test_r_group": v["test_r_group"],
                "test_R2": v["test_R2"]} for k, v in out.items()}

cv_records = []
for rep in range(N_REPEATS):
    seed = 1000 + rep
    res = run_one_split(seed)
    for model, metrics in res.items():
        cv_records.append({"repeat": rep, "seed": seed, "model": model, **metrics})
    print(f"repeat {rep+1}/{N_REPEATS} done")

df_cv = pd.DataFrame(cv_records)
df_cv.to_csv(TAB_DIR / "cv_robustness.csv", index=False)

summary_cv = df_cv.groupby("model")[["test_r_stacked", "test_r_group", "test_R2"]].agg(["mean", "std"])
print(summary_cv)


fig, ax = plt.subplots(figsize=(8, 5))
model_order = ["CP_baseline", "SC_spectral", "Geometric", "Hybrid"]
data_to_plot = [df_cv[df_cv["model"] == m]["test_r_stacked"].values for m in model_order]
bp = ax.boxplot(data_to_plot, labels=model_order, showmeans=True)
ax.set_ylabel(f"Test r (stacked), {N_REPEATS} random splits")
ax.set_title("C.3: Cross-split robustness of held-out performance")
ax.axhline(0, color="gray", lw=0.5)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_cv_robustness.png")
fig.savefig(FIG_DIR / "fig_cv_robustness.pdf")



mask_all = upper_tri(N)                    # every pair, regardless of SC distance
mask_d1  = upper_tri(N) & (Dmat == 1)      # direct structural neighbors only

pair_sensitivity_records = []
for label, mm_alt in [("all_pairs", mask_all), ("distance1_only", mask_d1),
                       ("distance_geq2_original", mask)]:
    print(f"\n=== Target set: {label} (n_pairs = {int(mm_alt.sum())}) ===")
    X_cp_alt  = cp_features_for_pairs(mm_alt)
    X_sc_alt  = spectral_features_for_pairs(eigvecs_sc, K_sc_best, mm_alt)
    X_geo_alt = spectral_features_for_pairs(eigvecs_geom, K_geo_best, mm_alt)

    res_alt = {}
    res_alt["CP_baseline"] = fit_predict(X_cp_alt, PLV_train, PLV_test, mm=mm_alt)
    res_alt["SC_spectral"] = fit_predict(X_sc_alt, PLV_train, PLV_test, mm=mm_alt)
    res_alt["Geometric"]   = fit_predict(X_geo_alt, PLV_train, PLV_test, mm=mm_alt)
    res_alt["Hybrid"]      = fit_predict(np.column_stack([X_cp_alt, X_sc_alt, X_geo_alt]),
                                          PLV_train, PLV_test, mm=mm_alt)

    for k, v in res_alt.items():
        print(f"  {k:14s} r_stacked={v['test_r_stacked']:.3f}  R2={v['test_R2']:.3f}")
        pair_sensitivity_records.append({
            "pair_set": label, "model": k, "n_pairs": int(mm_alt.sum()),
            "test_r_stacked": v["test_r_stacked"],
            "test_r_group": v["test_r_group"],
            "test_R2": v["test_R2"],
        })

df_pair_sens = pd.DataFrame(pair_sensitivity_records)
df_pair_sens.to_csv(TAB_DIR / "distant_pairs_sensitivity.csv", index=False)
print(df_pair_sens.round(4).to_string(index=False))


fig, ax = plt.subplots(figsize=(9, 5))
pair_sets = ["distance1_only", "distance_geq2_original", "all_pairs"]
x = np.arange(len(model_order))
width = 0.25
colors = {"distance1_only": "#e76f51", "distance_geq2_original": "#264653", "all_pairs": "#2a9d8f"}
for i, ps in enumerate(pair_sets):
    vals = [df_pair_sens[(df_pair_sens["pair_set"] == ps) & (df_pair_sens["model"] == m)]["test_r_stacked"].iloc[0]
            for m in model_order]
    ax.bar(x + (i-1)*width, vals, width=width, label=ps, color=colors[ps])
ax.set_xticks(x); ax.set_xticklabels(model_order, rotation=15, ha="right")
ax.set_ylabel("Held-out test r (stacked)")
ax.set_title("C.1: Sensitivity to distant-pairs restriction")
ax.legend(fontsize=8); ax.axhline(0, color="gray", lw=0.5); ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_distant_pairs_sensitivity.png")
fig.savefig(FIG_DIR / "fig_distant_pairs_sensitivity.pdf")




def build_consensus(threshold=None, weighted=False, mask_topology=None):
    if weighted:
        raw_w = np.log1p(SC_train.mean(axis=0))
        A_ = mask_topology.astype(bool).astype(np.float64) * raw_w
        np.fill_diagonal(A_, 0.0)
    else:
        A_ = ((SC_train > 0).mean(axis=0) >= threshold).astype(np.float64)
        np.fill_diagonal(A_, 0.0)
    return A_

configs = [
    ("binary_50pct",              dict(threshold=0.5,  weighted=False)),
    ("binary_60pct_original",     dict(threshold=0.6,  weighted=False)),
    ("binary_75pct",              dict(threshold=0.75, weighted=False)),
    ("weighted_masked_topology",  dict(threshold=None, weighted=True, mask_topology=A)),
    ("weighted_unrestricted_topology_FLAWED",
                                   dict(threshold=None, weighted=True,
                                        mask_topology=np.ones_like(A))),  # kept only for transparency
]

sc_sensitivity_records = []
for label, kwargs in configs:
    A_ = build_consensus(**kwargs)
    Abin_ = (A_ > 0).astype(np.int8)
    Dmat_ = shortest_path(Abin_, directed=False, unweighted=True)
    mask_ = upper_tri(N) & ((Dmat_ >= 2) | np.isinf(Dmat_))

    deg_ = A_.sum(1)
    with np.errstate(divide="ignore"):
        d12_ = np.where(deg_ > 0, 1.0/np.sqrt(deg_), 0.0)
    L_ = np.eye(N) - (d12_[:, None] * A_ * d12_[None, :])
    L_ = 0.5*(L_ + L_.T)
    _, eigvecs_ = np.linalg.eigh(L_)
    eigvecs_ = eigvecs_[:, 1:]

    cp_feats_ = {
        "common_neighbors":    common_neighbors(A_),
        "jaccard":             jaccard(A_),
        "adamic_adar":         adamic_adar(A_),
        "resource_allocation": resource_allocation(A_),
        "profile_similarity":  profile_similarity(A_),
        "communicability":     communicability(A_),
        "hub_affinity":        hub_affinity(A_),
        "neg_R_combinatorial": -effective_resistance(A_, normalized=False),
        "neg_R_symmetric":     -effective_resistance(A_, normalized=True),
    }
    X_cp_  = np.column_stack([P[mask_] for P in cp_feats_.values()])
    X_sc_  = spectral_features_for_pairs(eigvecs_, K_sc_best, mask_)
    X_geo_ = spectral_features_for_pairs(eigvecs_geom, K_geo_best, mask_)

    fits_ = {}
    fits_["CP_baseline"] = fit_predict(X_cp_, PLV_train, PLV_test, mm=mask_)
    fits_["SC_spectral"] = fit_predict(X_sc_, PLV_train, PLV_test, mm=mask_)
    fits_["Geometric"]   = fit_predict(X_geo_, PLV_train, PLV_test, mm=mask_)
    fits_["Hybrid"]      = fit_predict(np.column_stack([X_cp_, X_sc_, X_geo_]),
                                        PLV_train, PLV_test, mm=mask_)

    density = A_.astype(bool).sum() / (N * (N - 1))
    for m, v in fits_.items():
        sc_sensitivity_records.append({
            "sc_config": label, "model": m, "density": density,
            "n_pairs": int(mask_.sum()),
            "test_r_stacked": v["test_r_stacked"],
            "test_r_group": v["test_r_group"],
            "test_R2": v["test_R2"],
        })
    print(f"{label}: n_pairs={int(mask_.sum())}  done")

df_sc_sens = pd.DataFrame(sc_sensitivity_records)
df_sc_sens.to_csv(TAB_DIR / "sc_construction_sensitivity.csv", index=False)
print(df_sc_sens.round(4).to_string(index=False))


fig, ax = plt.subplots(figsize=(10, 5))
sc_configs_order = ["binary_50pct", "binary_60pct_original", "binary_75pct",
                     "weighted_masked_topology", "weighted_unrestricted_topology_FLAWED"]
width = 0.15
x = np.arange(len(model_order))
colors2 = {"binary_50pct": "#a8dadc", "binary_60pct_original": "#264653",
           "binary_75pct": "#e76f51", "weighted_masked_topology": "#2a9d8f",
           "weighted_unrestricted_topology_FLAWED": "#c9c9c9"}
for i, cfg in enumerate(sc_configs_order):
    vals = [df_sc_sens[(df_sc_sens["sc_config"] == cfg) & (df_sc_sens["model"] == m)]["test_r_stacked"].iloc[0]
            for m in model_order]
    ax.bar(x + (i - 2)*width, vals, width=width, label=cfg, color=colors2[cfg])
ax.set_xticks(x); ax.set_xticklabels(model_order, rotation=15, ha="right")
ax.set_ylabel("Held-out test r (stacked)")
ax.set_title("C.2: Sensitivity to SC consensus construction (topology-matched weighting)")
ax.legend(fontsize=7); ax.axhline(0, color="gray", lw=0.5); ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_sc_construction_sensitivity.png")
fig.savefig(FIG_DIR / "fig_sc_construction_sensitivity.pdf")
