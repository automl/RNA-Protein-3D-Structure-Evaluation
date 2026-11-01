"""
Sensitivity of the interface-placement classes to their thresholds.

evaluation/interface_rna_pose.py assigns every failed protein-RNA interface (DockQ < 0.23) to a
placement class with four thresholds: protein contact match for a correct site (0.1), protein
pocket RMSD for a mispredicted binding region (4 A), RNA translation for same place / shifted /
distal (10 / 20 A) and RNA rotation for a turned RNA (45 deg). Here each threshold is moved over a
range while the others stay at their defaults, and all of them are moved together over the full
grid, re-classifying the per-interface values of interface_rna_pose.tsv with the same rule
(interface_rna_pose.placement). Classes are merged as in the paper figure
(publication_figures/interface_failure_modes_figure.py). DockQ >= 0.23 (CAPRI acceptable) is not
varied: it defines success, not the failure mode.

Writes results/plots/all_chains/interface_rna_pose/placement_threshold_sensitivity.tsv
(one row per setting and period: share of the failures in each class) and prints a summary.
Run in the RNABench env:  python evaluation/placement_threshold_sensitivity.py
"""

import csv
import itertools
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import evaluation.interface_rna_pose as pose

TSV = next((p for p in (os.path.join(pose.OUTDIR, "interface_rna_pose.tsv"),
                        os.path.join(_REPO_ROOT, "results", "plots_odl", "all_chains", "interface_rna_pose",
                                     "interface_rna_pose.tsv")) if os.path.isfile(p)), None)
OUT = os.path.join(os.path.dirname(TSV), "placement_threshold_sensitivity.tsv")

# parameter: (module attribute, default, values tested)
PARAMS = {
    "protein_match": ("PROTEIN_SITE_MATCH", 0.1, [0.05, 0.1, 0.15, 0.2, 0.25]),
    "pocket_A": ("POCKET_A", 4.0, [3.0, 4.0, 5.0, 6.0]),
    "same_place_A": ("SAME_PLACE_A", 10.0, [7.5, 10.0, 12.5, 15.0]),
    "nearby_A": ("NEARBY_A", 20.0, [15.0, 20.0, 25.0, 30.0]),
    "turned_deg": ("TURNED_DEG", 45.0, [30.0, 45.0, 60.0, 90.0]),
}
MERGE = {"same place, same orientation": "right protein site, wrong RNA pose",
         "same place": "right protein site, wrong RNA pose"}
SHOWN = ["right protein site, wrong RNA pose", "different region", "protein binding region mispredicted",
         "shifted nearby", "same place, RNA turned", "same place, duplex slid"]
SHORT = {"right protein site, wrong RNA pose": "correct_site_incorrect_rna", "different region": "distal",
         "protein binding region mispredicted": "incorrect_binding_site", "shifted nearby": "shifted",
         "same place, RNA turned": "rotated", "same place, duplex slid": "slid_duplex"}


def _f(v):
    return float(v) if v not in ("", "nan", None) else np.nan


def load():
    rows = list(csv.DictReader(open(TSV), delimiter="\t"))
    for r in rows:
        for k in ("dockq", "protein_match", "pocket_rmsd", "rna_translation", "rna_rotation", "duplex_shift",
                  "duplex_axis_tilt"):
            r[k] = _f(r.get(k))
    return [r for r in rows if np.isfinite(r["dockq"])]


def _duplex(r, c):
    """a duplex at the native site with the wrong strand in place is placed by the whole duplex
    (evaluation/apply_duplex_pose.py), with the thresholds of the current setting"""
    if not np.isfinite(r["duplex_shift"]) or c in ("", "correct interface", "right protein site, wrong RNA pose",
                                                   "protein binding region mispredicted"):
        return c
    if r["duplex_shift"] <= pose.SAME_PLACE_A:
        return "same place, duplex slid" if r["duplex_axis_tilt"] <= pose.TURNED_DEG else "same place, RNA turned"
    return "shifted nearby" if r["duplex_shift"] <= pose.NEARBY_A else "different region"


def classify(rows, setting):
    for attr, val in setting.items():
        setattr(pose, attr, val)
    return [MERGE.get(c, c) for c in (_duplex(r, pose.placement(r["dockq"], r["rna_translation"], r["rna_rotation"],
                                                                r["protein_match"], r["pocket_rmsd"])) for r in rows)]


def shares(rows, classes, period):
    fail = [c for r, c in zip(rows, classes) if r["period"] == period and c not in ("correct interface", "")]
    n = len(fail)
    return n, {SHORT[k]: (100.0 * sum(c == k for c in fail) / n if n else np.nan) for k in SHOWN}


def main():
    rows = load()
    defaults = {attr: d for attr, d, _ in PARAMS.values()}
    base = classify(rows, defaults)
    stored = [MERGE.get(r["placement"], r["placement"]) for r in rows]
    mismatch = sum(a != b for a, b in zip(base, stored))
    print(f"{len(rows)} interfaces; default thresholds reproduce the stored classes "
          f"({mismatch} mismatches)")

    out = []
    # one threshold at a time
    for name, (attr, d, values) in PARAMS.items():
        for v in values:
            cl = classify(rows, {**defaults, attr: v})
            for period in ("post", "pre"):
                n, s = shares(rows, cl, period)
                out.append(dict(scan=name, setting=f"{name}={v:g}", is_default=v == d, period=period,
                                n_failures=n, **s))
    # all thresholds together (full grid)
    grid = []
    for combo in itertools.product(*[v for _, _, v in PARAMS.values()]):
        setting = {attr: v for (attr, _, _), v in zip(PARAMS.values(), combo)}
        cl = classify(rows, setting)
        n, s = shares(rows, cl, "post")
        grid.append(s)
        out.append(dict(scan="grid", setting=";".join(f"{k}={v:g}" for k, v in zip(PARAMS, combo)),
                        is_default=combo == tuple(d for _, d, _ in PARAMS.values()), period="post",
                        n_failures=n, **s))
    classify(rows, defaults)                         # restore the module defaults

    keys = ["scan", "setting", "is_default", "period", "n_failures"] + [SHORT[k] for k in SHOWN]
    with open(OUT, "w") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(keys)
        for r in out:
            w.writerow([f"{r[k]:.1f}" if isinstance(r[k], float) else r[k] for k in keys])

    print("\npost-cutoff failures, % per class (one threshold varied at a time):")
    hdr = "".join(f"{SHORT[k]:>27s}" for k in SHOWN)
    print(f"{'setting':24s}{'n':>5s}{hdr}")
    for r in out:
        if r["scan"] != "grid" and r["period"] == "post":
            print(f"{r['setting'] + (' *' if r['is_default'] else ''):24s}{r['n_failures']:5d}"
                  + "".join(f"{r[SHORT[k]]:27.1f}" for k in SHOWN))
    print(f"\nfull grid ({len(grid)} settings), post-cutoff failures: range [min, max] and median")
    for k in SHOWN:
        v = np.array([g[SHORT[k]] for g in grid])
        print(f"  {SHORT[k]:28s} {v.min():5.1f} - {v.max():5.1f}  (median {np.median(v):.1f})")
    lead = sum(max(g, key=g.get) == "correct_site_incorrect_rna" for g in grid)
    print(f"  correct site / incorrect RNA is the largest class in {lead}/{len(grid)} settings")
    print(f"\nWritten {OUT}")


if __name__ == "__main__":
    main()
