"""
Is a failed protein-RNA interface a ranking failure or a modelling failure?

For the 50 post-cutoff protein-RNA targets that were predicted with three seeds x five models
(data/all_data/other_analyses/seeds; the seed/model invariance analysis of the paper), every one of the 15 samples is
scored per protein-RNA interface with OpenStructure DockQ (local `ost compare-structures --dockq`,
ost conda env; same min-nuc-length 3 as the main analysis). Model 0 of seed 1 is the default
prediction used throughout the paper.

Per native protein-RNA interface:
  * top      DockQ of the default prediction (seed 1, model 0);
  * best5    best DockQ over the five models of seed 1 (what AF3 returned for that job);
  * best15   best DockQ over all 15 samples;
  * ranked   DockQ of the sample with the highest AF3 ranking_score over all 15 samples (how AF3
             would choose across seeds), and of the sample with the highest chain-pair ipTM of the
             interface (interface-specific confidence).
A failed default interface (DockQ < 0.23) that has an acceptable sample (>= 0.23) is a ranking
failure: AF3 generated a correct pose but did not put it first. Failures with no acceptable sample
among the 15 are modelling failures. Default-model failures are also split by their placement
class (interface_rna_pose.tsv), to ask whether misposed RNAs at the correct site are recoverable
by sampling.

Writes results/plots/all_chains/seed_sample_interfaces/{seed_sample_dockq.tsv,summary.txt}.
Run in the RNABench env (OST is called through the ost env binary):
    python evaluation/seed_sample_interfaces.py [--workers 8]
"""

import argparse
import csv
import glob
import json
import os
import sqlite3
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from evaluation.bundle_paths import ost_binary
OST = ost_binary()
from evaluation.bundle_paths import SEED_DB, seed_dir
REF_DIR = os.path.join(seed_dir(3), "pdb")
OUTDIR = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "seed_sample_interfaces")
JSON_DIR = os.path.join(OUTDIR, "ost_json")
# OST results of an earlier run (results/plots_odl) are reused rather than recomputed
JSON_DIRS = [JSON_DIR, os.path.join(_REPO_ROOT, "results", "plots_odl", "all_chains", "seed_sample_interfaces", "ost_json")]
POSE_TSV = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "interface_rna_pose", "interface_rna_pose.tsv")
ACCEPTABLE = 0.23
MERGE = {"same place, same orientation": "right protein site, wrong RNA pose"}


def targets():
    con = sqlite3.connect(SEED_DB[1])
    from evaluation.excluded_targets import POST_EXCLUDED     # 6FQL: released 2018, in AF3 training
    return sorted({r[0].strip().lower() for r in con.execute("SELECT exp_db_id FROM pred_protein_rna")}
                  - {x.lower() for x in POST_EXCLUDED})


def sample_files(pid):
    """[(seed, model, model cif, summary json)] for the 15 samples of one target."""
    out = []
    for seed in (1, 2, 3):
        for model in range(5):
            stem = f"fold_{pid}_s{seed}_model_{model}.cif"
            cif = os.path.join(seed_dir(seed), "af", stem)
            if not os.path.isfile(cif):
                continue
            summ = cif.replace(f"_model_{model}.cif", f"_summary_confidences_{model}.json")
            out.append((seed, model, cif, summ if os.path.exists(summ) else None))
    return out


def ost_json_path(pid, seed, model):
    """Cached OST JSON of one sample (None if not computed yet)."""
    for d in JSON_DIRS:
        p = os.path.join(d, f"{pid}_s{seed}_m{model}.json")
        if os.path.exists(p):
            return p
    return None


def run_ost(pid, seed, model, cif):
    path = ost_json_path(pid, seed, model) or os.path.join(JSON_DIR, f"{pid}_s{seed}_m{model}.json")
    if not os.path.exists(path):
        subprocess.run([OST, "compare-structures", "-m", cif, "-r", os.path.join(REF_DIR, f"{pid}.cif"),
                        "--dockq", "--min-nuc-length", "3", "-ft", "-o", path],
                       capture_output=True, text=True)
    return path


def chain_types(cif):
    import gemmi
    st = gemmi.read_structure(cif)
    st.setup_entities()
    out = {}
    for ch in st[0]:
        t = ch.get_polymer().check_polymer_type()
        out[ch.name] = ("rna" if t == gemmi.PolymerType.Rna else
                        "protein" if t in (gemmi.PolymerType.PeptideL, gemmi.PolymerType.PeptideD) else "other")
    return out


def model_chain_order(cif):
    order, cols, idx = [], [], None
    with open(cif) as fh:
        for line in fh:
            if line.startswith("_atom_site."):
                cols.append(line.strip().split(".", 1)[1])
            elif line.startswith(("ATOM", "HETATM")):
                idx = idx if idx is not None else cols.index("auth_asym_id")
                ch = line.split()[idx]
                if ch not in order:
                    order.append(ch)
    return order


def score_sample(pid, seed, model, cif, summ, fallback=None):
    """Protein-RNA rows of one sample: OST's DockQ, plus the exact-mapping DockQ of
    evaluation/interface_own_dockq.py for native interfaces OST did not score (poly-U RNAs, failed runs)."""
    js = json.load(open(run_ost(pid, seed, model, cif)))
    extra = (fallback or {}).get((pid.upper(), seed, model), [])
    if js.get("status") != "SUCCESS":
        js = {}
        if not extra:
            return []
    types = chain_types(cif)
    conf = json.load(open(summ)) if summ else {}
    order = model_chain_order(cif)
    ip = conf.get("chain_pair_iptm")
    rows = []
    for (r1, r2, m1, m2), dq in zip(js.get("dockq_interfaces", []), js.get("dockq", [])):
        if {types.get(m1), types.get(m2)} != {"protein", "rna"}:
            continue
        pair_ip = np.nan
        if ip and m1 in order and m2 in order:
            i, j = order.index(m1), order.index(m2)
            if i < len(ip) and j < len(ip[i]):
                pair_ip = ip[i][j]
        prot, rna = (m1, m2) if types[m1] == "protein" else (m2, m1)
        rows.append(dict(pdb=pid.upper(), seed=seed, model=model, ref_pair=":".join(sorted((r1, r2))),
                         mdl_pair=f"{prot}:{rna}", dockq=float(dq), dockq_source="ost",
                         ranking_score=conf.get("ranking_score", np.nan), pair_iptm=pair_ip))
    done = {r["ref_pair"] for r in rows}
    for ref_pair, mdl_pair, dq in extra:
        if ref_pair in done:
            continue
        prot, rna = mdl_pair.split(":")
        pair_ip = np.nan
        if ip and prot in order and rna in order:
            i, j = order.index(prot), order.index(rna)
            if i < len(ip) and j < len(ip[i]):
                pair_ip = ip[i][j]
        rows.append(dict(pdb=pid.upper(), seed=seed, model=model, ref_pair=ref_pair, mdl_pair=mdl_pair, dockq=dq,
                         dockq_source="own", ranking_score=conf.get("ranking_score", np.nan), pair_iptm=pair_ip))
    return rows


def _pose_classes():
    """{(pdb, model protein:rna pair): placement} for the default (seed 1, model 0) prediction."""
    out = {}
    if not os.path.exists(POSE_TSV):
        return out
    for r in csv.DictReader(open(POSE_TSV), delimiter="\t"):
        if r["period"] == "post":
            out[(r["pdb"].upper(), r["pred_pair"])] = MERGE.get(r["placement"], r["placement"])
    return out


def analyse(rows):
    by_if = defaultdict(list)
    for r in rows:
        by_if[(r["pdb"], r["ref_pair"])].append(r)
    pose = _pose_classes()
    recs = []
    for (pdb, ref_pair), rs in by_if.items():
        top = [r for r in rs if r["seed"] == 1 and r["model"] == 0]
        if not top:
            continue
        top = top[0]
        s1 = [r for r in rs if r["seed"] == 1]
        by_rank = max(rs, key=lambda r: r["ranking_score"] if np.isfinite(r["ranking_score"]) else -1)
        by_ip = max(rs, key=lambda r: r["pair_iptm"] if np.isfinite(r["pair_iptm"]) else -1)
        recs.append(dict(pdb=pdb, ref_pair=ref_pair, mdl_pair=top["mdl_pair"], n_samples=len(rs),
                         top=top["dockq"], best5=max(r["dockq"] for r in s1), best15=max(r["dockq"] for r in rs),
                         ranked=by_rank["dockq"], by_iptm=by_ip["dockq"],
                         n_acceptable=sum(r["dockq"] >= ACCEPTABLE for r in rs),
                         placement=pose.get((pdb, top["mdl_pair"]), "")))
    lines = []
    n = len(recs)
    ok = lambda k: sum(x[k] >= ACCEPTABLE for x in recs)
    lines.append(f"{n} protein-RNA interfaces in {len({x['pdb'] for x in recs})} targets, "
                 f"median {np.median([x['n_samples'] for x in recs]):.0f} samples each")
    for k, lab in (("top", "default (seed 1, model 0)"), ("best5", "best of seed-1 models"),
                   ("best15", "best of 15 samples"), ("ranked", "highest ranking_score of 15"),
                   ("by_iptm", "highest chain-pair ipTM of 15")):
        lines.append(f"  acceptable (DockQ >= {ACCEPTABLE}) with {lab:30s}: {ok(k):4d}/{n} ({100 * ok(k) / n:.0f}%), "
                     f"median DockQ {np.median([x[k] for x in recs]):.3f}")
    fails = [x for x in recs if x["top"] < ACCEPTABLE]
    rescued = [x for x in fails if x["best15"] >= ACCEPTABLE]
    lines.append(f"\ndefault-model failures: {len(fails)}; an acceptable sample exists for {len(rescued)} "
                 f"({100 * len(rescued) / max(len(fails), 1):.0f}%) -> ranking failures; "
                 f"{len(fails) - len(rescued)} modelling failures (no acceptable sample)")
    if rescued:
        lines.append(f"  among ranking failures, median share of acceptable samples "
                     f"{np.median([x['n_acceptable'] / x['n_samples'] for x in rescued]):.2f}; "
                     f"ranking_score picks an acceptable one in {sum(x['ranked'] >= ACCEPTABLE for x in rescued)}, "
                     f"pair ipTM in {sum(x['by_iptm'] >= ACCEPTABLE for x in rescued)}")
    lines.append("\nby placement class of the default-model failure:")
    for cls in sorted({x["placement"] for x in fails}):
        sub = [x for x in fails if x["placement"] == cls]
        r = sum(x["best15"] >= ACCEPTABLE for x in sub)
        lines.append(f"  {cls or '(no class)':40s} n={len(sub):3d}: acceptable sample exists {r} "
                     f"({100 * r / len(sub):.0f}%), median best-of-15 DockQ {np.median([x['best15'] for x in sub]):.3f}")
    with open(os.path.join(OUTDIR, "per_interface.tsv"), "w") as fh:
        keys = list(recs[0].keys()) if recs else []
        w = csv.writer(fh, delimiter="\t")
        w.writerow(keys)
        for x in recs:
            w.writerow([f"{x[k]:.3f}" if isinstance(x[k], float) else x[k] for k in keys])
    print("\n".join(lines))
    with open(os.path.join(OUTDIR, "summary.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(JSON_DIR, exist_ok=True)
    jobs = [(pid, *s) for pid in targets() for s in sample_files(pid)]
    print(f"{len(jobs)} samples for {len(targets())} targets", flush=True)
    from evaluation.interface_own_dockq import seed_fallback
    fb = seed_fallback()
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(lambda j: score_sample(*j, fallback=fb), jobs), 1):
            rows += r
            if i % 100 == 0:
                print(f"   scored {i}/{len(jobs)}", flush=True)
    keys = ["pdb", "seed", "model", "ref_pair", "mdl_pair", "dockq", "dockq_source", "ranking_score", "pair_iptm"]
    with open(os.path.join(OUTDIR, "seed_sample_dockq.tsv"), "w") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(keys)
        for r in rows:
            w.writerow([r[k] for k in keys])
    analyse(rows)


if __name__ == "__main__":
    main()
