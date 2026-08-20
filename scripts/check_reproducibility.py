"""
Reproducibility check: process one subject twice and compare.

Run this BEFORE launching the full dataset. It answers one question: does the
same subject give the same connectome on two runs? If not, the group results
would depend on when each subject happened to be processed.

Usage on Kaggle:
    import check_reproducibility as R
    R.check("sub-1093")
"""
import numpy as np
import dlbs_pipeline_v3 as P


def check(subject_id="sub-1093", config=P.Config):
    atlas = P.setup_atlas(config)
    template = P.setup_template(config, atlas=atlas)

    df = P.discover_subjects(
        config.FMRI_BASE, config.DTI_BASE, config.ANAT_BASE,
        batch_id=None, n_batches=None,
    )
    row = df[df["subject_id"] == subject_id].iloc[0]

    runs = []
    for i in (1, 2):
        print(f"\n{'=' * 60}\nRUN {i} of 2\n{'=' * 60}")
        # Force fresh registrations so we are testing the registration itself,
        # not the cache.
        import shutil
        shutil.rmtree(config.registration_dir(subject_id), ignore_errors=True)
        result = P.process_dti_subject(row, atlas, template, config)
        if not result["success"]:
            print(f"Run {i} failed: {result.get('error')}")
            return None
        runs.append(result)

    a, b = runs
    sc_a, sc_b = a["sc_matrix"], b["sc_matrix"]

    both = (sc_a > 0) | (sc_b > 0)
    agree = ((sc_a > 0) == (sc_b > 0))[both].mean() if both.any() else 0.0
    corr = np.corrcoef(sc_a[np.triu_indices_from(sc_a, 1)],
                       sc_b[np.triu_indices_from(sc_b, 1)])[0, 1]
    total_a, total_b = sc_a.sum(), sc_b.sum()

    qa, qb = a["qc"], b["qc"]
    print(f"\n{'=' * 60}\nREPRODUCIBILITY REPORT: {subject_id}\n{'=' * 60}")
    print(f"{'metric':32s} {'run 1':>14s} {'run 2':>14s}")
    for label, key in [
        ("atlas voxels (regions present)", "atlas_regions_present"),
        ("atlas volume ratio vs MNI", "atlas_volume_ratio_vs_mni"),
        ("streamlines kept", "n_streamlines_kept"),
        ("SC nonzero entries", "sc_nonzero"),
        ("SC total weight", "sc_total_weight"),
    ]:
        print(f"{label:32s} {qa[key]:>14} {qb[key]:>14}")
    print(f"\nEdge presence agreement : {100 * agree:.2f}%")
    print(f"Edge weight correlation : {corr:.4f}")
    print(f"Total weight difference : {100 * abs(total_a - total_b) / max(total_a, 1):.2f}%")
    print(f"ANTs seeded             : {qa['ants_seeded']}")

    identical = np.array_equal(sc_a, sc_b)
    print(f"SC matrices bit-identical: {identical}")

    # The weighted matrix can look stable while the metrics you actually
    # analyse are not, because they binarise and weak edges are the least
    # reproducible part. Measure that directly rather than assuming.
    iu = np.triu_indices_from(sc_a, 1)
    keys = ["common_neighbors", "jaccard", "adamic_adar",
            "resource_allocation", "communicability", "composite"]
    print(f"\n{'=' * 60}\nCOUPLING-POTENTIAL METRIC STABILITY\n{'=' * 60}")
    print(f"{'min streamlines':>15s} {'edges':>7s} " +
          " ".join(f"{k[:11]:>12s}" for k in keys))

    metric_rows = {}
    for thr in (1, 2, 3, 5, 10):
        ca = P.compute_coupling_potential_metrics(sc_a, min_streamlines=thr)
        cb = P.compute_coupling_potential_metrics(sc_b, min_streamlines=thr)
        row = []
        for k in keys:
            x, y = ca[k][iu], cb[k][iu]
            r = np.corrcoef(x, y)[0, 1] if x.std() > 0 and y.std() > 0 else float("nan")
            row.append(r)
        metric_rows[thr] = dict(zip(keys, row))
        n_edges = int((sc_a >= thr)[iu].sum())
        print(f"{thr:>15d} {n_edges:>7d} " + " ".join(f"{r:>12.3f}" for r in row))

    # Communicability is the only metric masked by (1 - A), so its zero pattern
    # IS the edge set. When the edge set disagrees between runs, the metric
    # looks unstable even if its values on the shared support are identical.
    # Correlating only over pairs that are non-edges in BOTH runs separates
    # "the metric is unstable" from "the support moved".
    ca = P.compute_coupling_potential_metrics(sc_a, min_streamlines=1)
    cb = P.compute_coupling_potential_metrics(sc_b, min_streamlines=1)
    shared = ((sc_a == 0) & (sc_b == 0))[iu]
    print(f"\nCommunicability restricted to pairs that are non-edges in both runs "
          f"({int(shared.sum())} of {len(iu[0])}):")
    if shared.sum() > 10:
        x, y = ca["communicability"][iu][shared], cb["communicability"][iu][shared]
        r_shared = np.corrcoef(x, y)[0, 1] if x.std() > 0 and y.std() > 0 else float("nan")
        r_all = np.corrcoef(ca["communicability"][iu], cb["communicability"][iu])[0, 1]
        print(f"  r over all pairs        : {r_all:.3f}")
        print(f"  r over shared non-edges : {r_shared:.3f}")
        if np.isfinite(r_shared) and r_shared - r_all > 0.15:
            print("  -> the metric itself is stable; the low overall number is the")
            print("     edge set moving, not the communicability values.")
    else:
        print("  too few shared non-edges to assess (the graph is very dense).")

    # Same thing with the standard, unmasked definition.
    ua = P.compute_coupling_potential_metrics(sc_a, min_streamlines=1,
                                              mask_existing_edges=False)
    ub = P.compute_coupling_potential_metrics(sc_b, min_streamlines=1,
                                              mask_existing_edges=False)
    x, y = ua["communicability"][iu], ub["communicability"][iu]
    r_unmasked = np.corrcoef(x, y)[0, 1] if x.std() > 0 and y.std() > 0 else float("nan")
    print(f"  r with CP_COMMUNICABILITY_MASK_EXISTING_EDGES=False : {r_unmasked:.3f}")

    # Reliability caps the effect sizes you can detect. A correlation between an
    # unreliable measure and something perfectly measured (age) is attenuated by
    # sqrt(reliability), and required sample size scales as 1/reliability.
    print(f"\n{'=' * 60}\nWHAT THIS MEANS FOR DETECTING AGE EFFECTS\n{'=' * 60}")
    print(f"{'metric':22s} {'reliability':>12s} {'attenuation':>12s} {'effective n of 170':>20s}")
    for k in keys:
        rel = metric_rows[1][k]
        if not np.isfinite(rel) or rel <= 0:
            continue
        print(f"{k:22s} {rel:>12.3f} {np.sqrt(rel):>12.3f} {170 * rel:>20.0f}")
    print("\nThis is PROCESSING reliability on identical input data, so it is a")
    print("best case. Scan-rescan reliability would be lower still.")

    density = float((sc_a[iu] > 0).mean())
    print(f"\nGraph density at SC_MIN_STREAMLINES=1: {100 * density:.1f}%")
    if density > 0.45:
        print("  That is high for a structural connectome (typically 10-30% after")
        print("  thresholding). Probabilistic tracking produces many weak, likely")
        print("  spurious edges. Consider a streamline or group-consistency")
        print("  threshold, which also happens to be the least reproducible part.")

    worst_at_1 = min(v for v in metric_rows[1].values() if np.isfinite(v))
    best_thr = max(
        metric_rows,
        key=lambda t: min(v for v in metric_rows[t].values() if np.isfinite(v)),
    )
    best_worst = min(v for v in metric_rows[best_thr].values() if np.isfinite(v))

    print(f"\n{'=' * 60}\nVERDICT\n{'=' * 60}")
    if identical:
        print("Fully reproducible. Safe to run the full dataset.")
    else:
        print(f"Weighted SC is {'stable' if corr > 0.97 else 'UNSTABLE'} "
              f"(r={corr:.3f}), but the binarised metrics you analyse "
              f"correlate as low as {worst_at_1:.3f} at SC_MIN_STREAMLINES=1.")
        if worst_at_1 > 0.9:
            print("That is acceptable. Proceed, and report the seeds in your methods.")
        else:
            print(f"Setting Config.SC_MIN_STREAMLINES = {best_thr} raises the worst "
                  f"metric to {best_worst:.3f}.")
            print("Choose a threshold on this evidence before processing the cohort,")
            print("or export DLBS_DETERMINISTIC=1 for bit-identical (slower) runs.")

    return {"corr": corr, "agreement": agree, "identical": identical,
            "metric_stability": metric_rows}
