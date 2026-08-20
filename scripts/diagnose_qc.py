"""
Separate bad-warp subjects from genuine coverage loss, iterating until the
answer stops changing, and write an exclusion list.

The problem this solves. Some parcels end up with no connections. There are
two causes with opposite remedies:

  A. FIELD OF VIEW. The acquisition never covered that anatomy. The same
     parcels are missing even in subjects whose registration is fine, so you
     drop those parcels for the whole cohort.

  B. REGISTRATION. The warp for a subject is off, and the peripheral parcels
     fall out first. You drop those subjects, not the parcels.

They are hard to tell apart in one pass, because bad-warp subjects make the
same peripheral parcels look systematically missing. So this iterates:

    pass 0  exclude on warp scale and gross failure only, never on isolated
            parcels, so the parcel statistics are not yet shaped by the rule
            that depends on them
    pass 1  among the survivors, find parcels still isolated often. Those are
            genuine coverage loss, and get excused
    pass 2  re-score subjects, counting only isolations OUTSIDE the excused
            set, and exclude those that still fail
    ...     repeat until the kept set and the excused set both stop changing
            (usually 2-3 passes)

Usage:

    import diagnose_qc as D
    result = D.run()                                        # iterates
    result = D.run(label_path="/path/Schaefer...order.txt")  # real names
    result = D.run(refine=False)                            # single pass
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path(os.getenv("DLBS_CONNECTOME_DIR", "./results"))

# Used when DATA_ROOT is read-only, e.g. once a merged dataset has been
# published and re-attached read-only (as on Kaggle, under /kaggle/input).
WRITABLE_FALLBACK = Path(os.getenv("DLBS_CONNECTOME_DIR", "./results"))

# A parcel isolated in at least this fraction of the SURVIVING subjects is
# genuine coverage loss.
SYSTEMATIC_FRACTION = 0.10

# Any isolated parcel outside the excused set means a region received no
# streamline endpoints. 0 is the strict reading; the tool prints what each
# choice costs before you commit.
MAX_ISOLATED_PARCELS = 0

# Hard bounds, for gross failures only. The modal warp band is learned from
# the data and is usually tighter than this.
VOLUME_RATIO_RANGE = (0.45, 1.60)
MIN_EDGE_DENSITY = 0.15
MAX_MEAN_FD_MM = 0.5

# Learn the warp band from the cohort itself. Percentiles of the
# zero-isolated-parcel subset look appealing but are biased: when coverage
# loss is present, that subset is exactly the subjects who happened to dodge
# it, and the band comes out far too tight. Median +/- k*MAD over ALL
# subjects is robust to the 20-30% that are genuinely off and does not depend
# on the isolation counts at all.
USE_MODAL_BAND = True
MODAL_BAND_MAD = 4.0   # lower is stricter; 4.0 is deliberately permissive

MAX_PASSES = 6
RATIO_COL = "dti_atlas_volume_ratio_vs_mni"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load(data_root=DATA_ROOT):
    data_root = Path(data_root)
    group = data_root / "group"
    subjects = (group / "subjects.txt").read_text().split()
    sc = np.load(group / "sc_count.npy")
    qc = pd.read_csv(data_root / "qc" / "qc_table.csv", index_col=0).reindex(subjects)
    return subjects, sc, qc


def parcel_labels(data_root=DATA_ROOT, n_parcels=100, label_path=None):
    """
    Schaefer names, if the label file can be found.

    The label file is an INPUT to the pipeline, not one of its outputs, so it
    is usually not inside the merged dataset. Names are cosmetic: everything
    downstream keys off parcel INDEX.
    """
    search = []
    if label_path:
        search.append(Path(label_path))
    for root in (Path(data_root), Path("/kaggle/input")):
        if root.exists():
            try:
                search.extend(sorted(root.rglob("Schaefer2018_100Parcels*order.txt")))
            except OSError:
                pass

    for candidate in search:
        try:
            names = []
            for line in Path(candidate).read_text().splitlines():
                parts = line.split()
                if len(parts) >= 2 and parts[0].isdigit():
                    names.append(parts[1])
            if len(names) == n_parcels:
                return np.array(names), str(candidate)
        except OSError:
            continue
    return np.array([f"parcel_{i + 1}" for i in range(n_parcels)]), None


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def _modal_band(ratio, per_subject_isolated=None):
    """
    Robust warp-scale band: median +/- MODAL_BAND_MAD scaled MADs.

    Uses every subject, so a cohort with widespread coverage loss does not
    shrink the band onto an unrepresentative subset.
    """
    finite = np.isfinite(ratio)
    if finite.sum() < 20:
        return None
    median = float(np.median(ratio[finite]))
    mad = float(np.median(np.abs(ratio[finite] - median)))
    if mad <= 0:
        return None
    spread = MODAL_BAND_MAD * 1.4826 * mad
    return float(median - spread), float(median + spread)


def _score(subjects, sc, qc, isolated, excused, band, use_isolated_rule):
    """
    Decide who to exclude. `excused` is the boolean parcel mask of genuine
    coverage loss, which does not count against a subject.
    """
    reasons = {s: [] for s in subjects}
    counts = isolated[:, ~excused].sum(axis=1)

    ratio = (qc[RATIO_COL].values.astype(float)
             if RATIO_COL in qc.columns else np.full(len(subjects), np.nan))

    lo_hard, hi_hard = VOLUME_RATIO_RANGE
    for i, subject in enumerate(subjects):
        if np.isfinite(ratio[i]) and not (lo_hard <= ratio[i] <= hi_hard):
            reasons[subject].append(
                f"warp ratio {ratio[i]:.3f} outside [{lo_hard}, {hi_hard}]")
        elif band and np.isfinite(ratio[i]) and not (band[0] <= ratio[i] <= band[1]):
            reasons[subject].append(
                f"warp ratio {ratio[i]:.3f} outside the modal band "
                f"[{band[0]:.2f}, {band[1]:.2f}]")

        if (sc[i] > 0).sum() == 0:
            reasons[subject].append("SC matrix is entirely empty")

    if "sc_edge_density" in qc.columns:
        dens = qc["sc_edge_density"].values.astype(float)
        for i, subject in enumerate(subjects):
            if np.isfinite(dens[i]) and dens[i] < MIN_EDGE_DENSITY:
                reasons[subject].append(
                    f"SC edge density {dens[i]:.3f} below {MIN_EDGE_DENSITY}")

    if use_isolated_rule:
        for i, subject in enumerate(subjects):
            if counts[i] > MAX_ISOLATED_PARCELS:
                reasons[subject].append(
                    f"{int(counts[i])} isolated parcels outside the excused set")

    return {s: r for s, r in reasons.items() if r}, counts


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def _writable_target(data_root, out_root=None):
    """
    Where to write exclusions.json.

    Once the merged dataset is published as a Kaggle dataset it is mounted
    read-only under /kaggle/input, so writing next to it fails. Fall back to
    the working directory, which the notebook also searches.
    """
    candidates = []
    if out_root:
        candidates.append(Path(out_root) / "qc")
    candidates.append(Path(data_root) / "qc")
    candidates.append(Path(WRITABLE_FALLBACK) / "qc")
    candidates.append(Path.cwd() / "qc")

    for folder in candidates:
        try:
            folder.mkdir(parents=True, exist_ok=True)
            probe = folder / ".write_test"
            probe.write_text("ok")
            probe.unlink()
            return folder / "exclusions.json"
        except OSError:
            continue
    raise OSError("No writable location found for exclusions.json. "
                  "Pass out_root= explicitly.")


def run(data_root=DATA_ROOT, verbose=True, label_path=None, refine=True,
        max_passes=MAX_PASSES, out_root=None):
    subjects, sc, qc = load(data_root)
    n_subjects, n_parcels = sc.shape[0], sc.shape[1]
    names, label_src = parcel_labels(data_root, n_parcels, label_path)

    isolated = (sc > 0).sum(axis=2) == 0          # (subjects, parcels)
    per_subject_all = isolated.sum(axis=1)

    def say(text=""):
        if verbose:
            print(text)

    say("=" * 72)
    say(f"ISOLATED PARCELS  ({n_subjects} subjects, {n_parcels} parcels)")
    say("=" * 72)
    if label_src is None:
        say("NOTE: Schaefer label file not found, so parcels are shown by index")
        say("      (parcel_N means index N-1). Pass label_path= for real names.")
    else:
        say(f"Labels from {label_src}")
    say(f"Subjects with at least one isolated parcel: "
        f"{int((per_subject_all > 0).sum())} "
        f"({100 * (per_subject_all > 0).mean():.0f}%)")
    say(f"Total isolations across the cohort: {int(isolated.sum())}")

    ratio = (qc[RATIO_COL].values.astype(float)
             if RATIO_COL in qc.columns else np.full(n_subjects, np.nan))
    band = _modal_band(ratio) if USE_MODAL_BAND else None
    if band:
        n_out = int(np.sum(np.isfinite(ratio) & ((ratio < band[0]) | (ratio > band[1]))))
        say(f"\nModal warp band (median +/- {MODAL_BAND_MAD} MADs over all subjects):")
        say(f"  {band[0]:.3f} to {band[1]:.3f}   "
            f"(cohort median {np.nanmedian(ratio):.3f}, "
            f"range {np.nanmin(ratio):.3f} to {np.nanmax(ratio):.3f})")
        say(f"  subjects outside it: {n_out}")

    # ---------------- iterate ----------------
    excused = np.zeros(n_parcels, dtype=bool)
    history = []
    keep = list(subjects)

    say("\n" + "-" * 72)
    say("Iterating")
    say("-" * 72)

    for p in range(max_passes):
        # Pass 0 deliberately ignores the isolated-parcel rule, so the parcel
        # statistics computed on the survivors are not shaped by the very rule
        # that depends on them.
        use_isolated_rule = p > 0
        exclude, _ = _score(subjects, sc, qc, isolated, excused, band,
                            use_isolated_rule)
        keep_new = [s for s in subjects if s not in exclude]
        keep_idx = np.array([i for i, s in enumerate(subjects) if s not in exclude])

        if len(keep_idx) == 0:
            say("Every subject was excluded. Loosen the thresholds.")
            break

        per_parcel_keep = isolated[keep_idx].sum(axis=0)
        excused_new = per_parcel_keep >= SYSTEMATIC_FRACTION * len(keep_idx)

        history.append({
            "pass": p,
            "used_isolated_rule": bool(use_isolated_rule),
            "n_kept": len(keep_new),
            "n_excused_parcels": int(excused_new.sum()),
            "excused_indices": np.flatnonzero(excused_new).tolist(),
        })

        say(f"  pass {p}: kept {len(keep_new)}/{n_subjects}, "
            f"parcels excused as coverage loss: {int(excused_new.sum())}"
            + (f"  {names[excused_new].tolist()}" if excused_new.any() else ""))

        converged = (keep_new == keep) and np.array_equal(excused_new, excused)
        keep, excused = keep_new, excused_new

        if not refine:
            exclude, _ = _score(subjects, sc, qc, isolated, excused, band, True)
            keep = [s for s in subjects if s not in exclude]
            say("  (refine=False: stopping after one scoring pass)")
            break
        if converged and p > 0:
            say(f"  converged after {p + 1} passes")
            break
    else:
        say(f"  stopped at {max_passes} passes without full convergence; "
            "using the last state")

    keep_set = set(keep)
    keep_idx = np.array([i for i, s in enumerate(subjects) if s in keep_set])
    per_parcel_keep = isolated[keep_idx].sum(axis=0)

    # ---------------- what the survivors say ----------------
    say("\n" + "-" * 72)
    say("Which cause is it?")
    say("-" * 72)
    total = int(isolated.sum())
    dropped_idx = [i for i, s in enumerate(subjects) if s not in keep_set]
    from_excluded = int(isolated[dropped_idx].sum()) if dropped_idx else 0
    pct_excluded = 100 * from_excluded / total if total else 0.0
    say(f"Isolations contributed by the EXCLUDED subjects: {pct_excluded:.0f}%")
    say(f"Isolations remaining among the {len(keep)} survivors: "
        f"{int(isolated[keep_idx].sum())}")

    if excused.any():
        say(f"\nParcels still isolated in >= {100 * SYSTEMATIC_FRACTION:.0f}% of "
            f"survivors, i.e. genuine coverage loss:")
        say(f"{'parcel':<34s} {'survivors affected':>19s}")
        for idx in np.flatnonzero(excused)[np.argsort(-per_parcel_keep[excused])]:
            say(f"{names[idx]:<34s} {int(per_parcel_keep[idx]):>12d} "
                f"({100 * per_parcel_keep[idx] / len(keep):.1f}%)")
        say("\n-> Set DROP_SYSTEMATIC_PARCELS = True in the notebook.")
    else:
        say("\nNo parcel is isolated in a meaningful share of the survivors, so")
        say("there is no genuine coverage loss: the isolations came from the")
        say("excluded subjects' registrations.")
        say("\n-> Keep all parcels. DROP_SYSTEMATIC_PARCELS = False.")

    if pct_excluded > 80 and not excused.any():
        say("\nVERDICT: registration, not coverage.")
    elif excused.any() and pct_excluded < 50:
        say("\nVERDICT: mostly genuine coverage loss.")
    elif excused.any():
        say("\nVERDICT: both causes present. The excluded subjects account for")
        say("most isolations, but some parcels stay uncovered in good subjects.")

    # ---------------- costs ----------------
    say("\n" + "-" * 72)
    say("What each rule would cost")
    say("-" * 72)
    counts_excused = isolated[:, ~excused].sum(axis=1)
    say(f"{'rule':<48s} {'kept':>6s}")
    say(f"{'no exclusions':<48s} {n_subjects:>6d}")
    for k in (0, 1, 3, 5, 10):
        label = ("no isolated parcels outside the excused set" if k == 0
                 else f"at most {k} isolated parcels outside it")
        say(f"{label:<48s} {int((counts_excused <= k).sum()):>6d}")
    if band:
        in_band = np.isfinite(ratio) & (ratio >= band[0]) & (ratio <= band[1])
        say(f"{f'warp ratio inside {band[0]:.2f} to {band[1]:.2f}':<48s} "
            f"{int(in_band.sum()):>6d}")
    say(f"{'BOTH (what this run applied)':<48s} {len(keep):>6d}")

    # ---------------- final exclusion list ----------------
    exclude, _ = _score(subjects, sc, qc, isolated, excused, band, True)
    fmri_exclude = {}
    if "fmri_mean_fd_mm" in qc.columns:
        for subject, value in qc["fmri_mean_fd_mm"].items():
            if np.isfinite(value) and value > MAX_MEAN_FD_MM:
                fmri_exclude[subject] = f"mean FD {value:.3f} mm above {MAX_MEAN_FD_MM}"

    say("\n" + "=" * 72)
    say("FINAL EXCLUSIONS")
    say("=" * 72)
    say(f"Structural (drop everywhere): {len(exclude)}")
    for subject, why in sorted(exclude.items()):
        say(f"  {subject}: {'; '.join(why)}")
    say(f"\nFunctional only (keep SC, drop FC): {len(fmri_exclude)}")
    for subject, why in sorted(fmri_exclude.items()):
        say(f"  {subject}: {why}")
    say(f"\nSurviving for SC analyses: {len(keep)} of {n_subjects}")

    # ---------------- concentrated in one batch? ----------------
    # If one processing batch or acquisition wave contributes most of the
    # failures, the cause is upstream and fixable, not per-subject noise.
    if exclude:
        say("\n" + "-" * 72)
        say("Is exclusion concentrated anywhere?")
        say("-" * 72)

        groups = {}
        if "part" in qc.columns:
            groups["processing batch"] = qc["part"].astype(str).to_dict()
        # DLBS-style ids come in different lengths across collection waves.
        id_len = {s: f"{len(str(s).split('-')[-1])}-digit id" for s in subjects}
        if len(set(id_len.values())) > 1:
            groups["subject id format"] = id_len

        for label, mapping in groups.items():
            say(f"\nBy {label}:")
            say(f"{'group':<22s} {'n':>6s} {'excluded':>10s} {'rate':>8s}")
            rates = {}
            for value in sorted({v for v in mapping.values() if v is not None}):
                members = [s for s in subjects if mapping.get(s) == value]
                if not members:
                    continue
                n_ex = sum(1 for s in members if s in exclude)
                rates[value] = n_ex / len(members)
                say(f"{str(value):<22s} {len(members):>6d} {n_ex:>10d} "
                    f"{100 * rates[value]:>7.1f}%")
            # A 30-point spread across a group of 16 is unremarkable. Test it
            # rather than flagging on the raw difference, which cries wolf.
            table = []
            for value in rates:
                members = [s for s in subjects if mapping.get(s) == value]
                n_ex = sum(1 for s in members if s in exclude)
                table.append([n_ex, len(members) - n_ex])
            table = np.array(table)

            pvalue = None
            try:
                from scipy.stats import chi2_contingency, fisher_exact

                usable = table[table.sum(axis=1) >= 5]
                if usable.shape[0] == 2:
                    pvalue = float(fisher_exact(usable)[1])
                elif usable.shape[0] > 2:
                    pvalue = float(chi2_contingency(usable.T)[1])
            except Exception:
                pvalue = None

            spread = max(rates.values()) - min(rates.values())
            if pvalue is not None:
                say(f"  spread {100 * spread:.0f} points, p = {pvalue:.3f}")
                if pvalue < 0.05:
                    say("  -> real concentration. Looks like a batch or acquisition")
                    say("     effect rather than per-subject noise. Worth chasing")
                    say("     upstream before accepting the losses.")
                else:
                    say("  -> consistent with chance at these group sizes. No action.")
            elif spread > 0.20:
                say(f"  spread {100 * spread:.0f} points, groups too small to test.")

    # ---------------- bias warning ----------------
    if RATIO_COL in qc.columns and exclude:
        kept_ratio = ratio[[i for i, s in enumerate(subjects) if s in keep_set]]
        drop_ratio = ratio[[i for i, s in enumerate(subjects) if s in exclude]]
        say("\n" + "-" * 72)
        say("Check this before you use the list")
        say("-" * 72)
        say(f"warp ratio, kept     : median {np.nanmedian(kept_ratio):.3f}")
        say(f"warp ratio, excluded : median {np.nanmedian(drop_ratio):.3f}")
        k_med = float(np.nanmedian(kept_ratio))
        d_med = float(np.nanmedian(drop_ratio))
        say("")
        if d_med > k_med:
            say("The excluded group skews to HIGH ratios, i.e. an over-expanded")
            say("warp, which corresponds to an apparently LARGER brain. Atrophy")
            say("runs the other way, so this may remove younger subjects rather")
            say("than older ones.")
        else:
            say("The excluded group skews to LOW ratios, i.e. a smaller brain")
            say("relative to the template, which is what advanced atrophy looks")
            say("like. This may preferentially remove your oldest subjects and")
            say("shrink the effect you are measuring.")
        say("Either way, test age balance between kept and excluded, and report")
        say("the analysis both ways.")

    payload = {
        "n_subjects": int(n_subjects),
        "n_parcels": int(n_parcels),
        "modal_band": list(band) if band else None,
        "systematic_parcels": names[excused].tolist(),
        "systematic_parcel_indices": np.flatnonzero(excused).tolist(),
        "drop_systematic_parcels_recommended": bool(excused.any()),
        "exclude_structural": exclude,
        "exclude_functional": fmri_exclude,
        "keep": keep,
        "passes": history,
        "thresholds": {
            "max_isolated_parcels": MAX_ISOLATED_PARCELS,
            "volume_ratio_range": list(VOLUME_RATIO_RANGE),
            "min_edge_density": MIN_EDGE_DENSITY,
            "max_mean_fd_mm": MAX_MEAN_FD_MM,
            "systematic_fraction": SYSTEMATIC_FRACTION,
            "use_modal_band": bool(USE_MODAL_BAND),
        },
    }
    target = _writable_target(data_root, out_root)
    target.write_text(json.dumps(payload, indent=2))
    say(f"\nWritten to {target}")
    if Path(target).parent.parent != Path(data_root):
        say(f"(the dataset at {data_root} is read-only, so this went to a")
        say(" writable location instead; the notebook checks both)")
    say("Edit the thresholds at the top of this file and re-run if you disagree.")

    return {
        "subjects": subjects,
        "keep": keep,
        "exclude_structural": exclude,
        "exclude_functional": fmri_exclude,
        "excused_parcel_indices": np.flatnonzero(excused).tolist(),
        "excused_parcel_names": names[excused].tolist(),
        "isolated_per_subject": pd.Series(per_subject_all, index=subjects),
        "per_parcel_survivors": pd.Series(per_parcel_keep, index=names),
        "passes": history,
        "modal_band": band,
    }


if __name__ == "__main__":
    run()
