"""
Scored framework v2 (decision_tree.framework) applied to the binding-specificity sets.

These sets have no deposited structure (RNAcompete) or score sequence specificity (CLIP), so they
do not calibrate the tree. They ask whether AF3's binding-preference signal holds up better where
the framework rates the predicted interface as more reliable.

  v4    RNAcompete single-probe complexes (data/rnaCompete_v4/rnaCompete_v4.db): interface leaf of
        every probe complex; per-protein maxprobe-vs-minprobe AUC by how many probes pass the
        confidence node.
  v5    RNAcompete competition complexes (data/rnaCompete_v5_MINMAX/rna_compete_minmax.db): leaf of
        every probe chain; P(maxprobe ranks first) and competition AUC by the best chain's leaf.
  CLIP  motif-atlas footprints of deposited structures (clip/results/positional.tsv): positional
        recovery of the motif-bearing footprint by the best RNA chain's leaf
        (answers from decision_tree.validate_tree).

Writes results/plots/all_chains/decision_tree/applications_tree_*.tsv and
applications_tree_summary.txt. Run from the repo root (RNABench env):
    python -m decision_tree.applications          # v2 (paper)
    python -m decision_tree.applications --v3     # decision tree v3: v2 interface side + v3 fold side
    python -m decision_tree.applications --probability   # calibrated probabilities (paper)
"""

from __future__ import annotations

import contextlib
import csv
import io
import logging
import os
import sys
from collections import Counter

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "rnaCompete")):
    if p not in sys.path:
        sys.path.insert(0, p)


DATA = os.path.join(_REPO_ROOT, "data")
def _first(*paths):
    """the published layout (data/rnacompete/, fetch_data.sh; CSV from rnaCompete/prep_rnacompete_dbs.py) first,
    else the processing layout"""
    return next((p for p in paths if os.path.exists(p)), paths[-1])


V4_DB = _first(os.path.join(DATA, "rnacompete", "rnaCompete_v4.db"), os.path.join(DATA, "rnaCompete_v4", "rnaCompete_v4.db"))
V5_DB = _first(os.path.join(DATA, "rnacompete", "rnaCompete_v5_MINMAX.db"),
               os.path.join(DATA, "rnaCompete_v5_MINMAX", "rna_compete_minmax.db"))
V5_CSV = _first(os.path.join(DATA, "rnacompete", "exp_protein_rna.csv"),
                os.path.join(DATA, "rnaCompete_v5_MINMAX", "protein_rna", "exp_protein_rna.csv"))
CLIP = os.path.join(_REPO_ROOT, "clip", "results")
OUTDIR = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "decision_tree")


# =========================================================================== #
# Scored framework v2 (decision_tree.framework): interface levels
# =========================================================================== #
from decision_tree import framework as fw
from decision_tree.rules import interface_tier
from decision_tree.validate import _auc

LEVELS = fw.INTERFACE_LEVELS
RANK = {l: i for i, l in enumerate(LEVELS)}


def _share(levels):
    c = Counter(l for l in levels if l)
    n = sum(c.values())
    return ", ".join(f"{l} {100 * c[l] / n:.0f}%" for l in LEVELS if c[l]) + f" (n={n})" if n else "n=0"


def _domains_by(db, key):
    import sqlite3
    out = {}
    for pid, names in sqlite3.connect(db).execute("SELECT PDBId, domain_names FROM exp_protein_rna"):
        d = [x for x in str(names or "").split(";") if x.strip() not in ("", "-")]
        if d:
            out.setdefault(key(pid), d)
    return out


def _scalar(v):
    import json
    try:
        v = json.loads(v) if isinstance(v, str) and v.strip().startswith("[") else v
        v = v[0] if isinstance(v, list) and v else v
        return float(v)
    except (TypeError, ValueError, IndexError):
        return None


def _level(iptm, pae, domains, ptm, plddt, familiar=False):
    """RNAcompete probes are synthetic RNAs: not familiar."""
    return fw.interface_level(rna_iptm=iptm, pair_pae=pae, familiar=familiar, domains=domains,
                              partner_ptm=ptm, rna_plddt=plddt)


def v2_rnacompete_v4(L):
    import probe_ranking_within_protein as pr
    proteins = pr.load_proteins(V4_DB)
    doms = _domains_by(V4_DB, lambda p: str(p).strip().upper())
    rows = []
    for p in proteins:
        for probe in p["maxprobes"] + p["minprobes"]:
            f = probe["features"]
            ptm = f["af3_protein_pTM"] if f["af3_protein_pTM"] == f["af3_protein_pTM"] else None
            probe["level"] = _level(probe["raw"]["af3_rna_ipTM"], f["af3_chain_pair_pae_min"],
                                    doms.get(str(probe["pdb_id"]).strip().upper()), ptm, f["af3_rna_pLDDT_avg"])
            rows.append(dict(protein=p["protein"], probe=probe["pdb_id"], cls=probe["class"], level=probe["level"]))
        p["n_ok"] = sum(pp["level"] in ("CAUTION", "RELIABLE") for pp in p["maxprobes"] + p["minprobes"])
        p["n_probes"] = len(p["maxprobes"]) + len(p["minprobes"])
    mx = [r["level"] for r in rows if r["cls"] == "maxprobe"]
    mn = [r["level"] for r in rows if r["cls"] == "minprobe"]
    L.append(f"v4 RNAcompete single-probe complexes ({len(proteins)} RBPs): interface level, maxprobes {_share(mx)}")
    L.append(f"  minprobes {_share(mn)}; AUC(level, maxprobe)="
             f"{_auc([RANK[l] for l in mx if l], [RANK[l] for l in mn if l]):.3f}")
    for name, f in (("all probes CAUTION or better", lambda p: p["n_ok"] == p["n_probes"]),
                    ("some", lambda p: 0 < p["n_ok"] < p["n_probes"]),
                    ("none (all UNRELIABLE)", lambda p: p["n_ok"] == 0)):
        sub = [p for p in proteins if f(p)]
        if sub:
            parts = []
            for key, lab in (("af3_rna_ipTM", "ipTM_r"), ("RNAmotif_score", "motif score")):
                acc = list(pr.rank_accuracy(sub, key).values())
                parts.append(f"{lab} per-protein AUC {np.mean(acc):.3f}" if acc else f"{lab} n/a")
            L.append(f"  RBPs with {name:30s} n={len(sub):3d}: " + "; ".join(parts))
    return rows


def v2_rnacompete_v5(L):
    import sqlite3
    import exp_probeMax_combined as ec
    with contextlib.redirect_stdout(io.StringIO()):
        recs = ec.build_competition_records(ec.load_pred_rows(V5_DB), ec.load_rna_exp_map(V5_CSV),
                                            ec.load_domain_family_map(V5_DB))
    doms = _domains_by(V5_DB, lambda p: ec.extract_base_pdb_id(p))
    meta = {eid: (ptm, pl) for eid, ptm, pl in sqlite3.connect(V5_DB).execute(
        "SELECT exp_db_id, af3_protein_pTM, af3_rna_pLDDT_avg FROM pred_protein_rna")}
    rows = []
    for r in recs:
        ptm, pl = (_scalar(x) for x in meta.get(r["pdb_id"], (None, None)))
        d = doms.get(ec.extract_base_pdb_id(r["pdb_id"]))
        for c in r["chains"]:
            pae = -c["pae_oriented"] if np.isfinite(c.get("pae_oriented", np.nan)) else None
            c["level"] = _level(c.get("rna_ipTM"), pae, d, ptm, pl)
            rows.append(dict(complex=r["pdb_id"], role=c["role"], level=c["level"]))
        ranks = [RANK[c["level"]] for c in r["chains"] if c["level"]]
        r["best"] = max(ranks) if ranks else None
    mx = [x["level"] for x in rows if x["role"] == "maxprobe"]
    mn = [x["level"] for x in rows if x["role"] == "minprobe"]
    L.append(f"\nv5 RNAcompete competition complexes ({len(recs)}): interface level, maxprobe chains {_share(mx)}")
    L.append(f"  minprobe chains {_share(mn)}; AUC(level, maxprobe)="
             f"{_auc([RANK[l] for l in mx if l], [RANK[l] for l in mn if l]):.3f}")
    metrics = (("rna_ipTM", "ipTM_r"), ("RNAmotif_score_rank", "motif score"))
    for name, f in (("best chain UNRELIABLE", lambda v: v == 0), ("best chain CAUTION or better", lambda v: v >= 1)):
        sub = [r for r in recs if r["best"] is not None and f(r["best"])]
        if not sub:
            continue
        wr = ec.competition_win_rates(sub, tuple((k, lab, "") for k, lab in metrics))
        au = ec.competition_auc(sub, tuple((k, lab, "") for k, lab in metrics))
        L.append(f"  {name:30s} n={len(sub):3d}: " + "; ".join(
            f"{lab} P(first) {wr[k]['rate']:.2f} AUC {au[k]['auc'] if au[k]['auc'] is not None else float('nan'):.3f}"
            for k, lab in metrics))
    return rows


def v2_clip(L):
    from decision_tree.develop import interface_rows
    sel = fw.calibration().get("design", {}).get("interface_downgrades", [])
    best = {}
    for r in interface_rows():
        r["low_ptm_or_plddt"] = r["wellfolded"] == "False"
        if r["familiar"] is None:
            continue
        sc = r["tier"] - sum(r[m] for m in sel) + int(r["familiar"])
        lv = 0 if sc <= 1 else (1 if sc == 2 else 2)
        k = r["pdb"].upper()
        best[k] = max(best.get(k, -1), lv)
    pos = list(csv.DictReader(open(os.path.join(CLIP, "positional.tsv")), delimiter="\t"))
    rows = []
    for r in pos:
        lv = best.get(r["pdb"].upper())
        f = lambda k: float(r[k]) if r.get(k) not in ("", "nan", None) else np.nan
        rows.append(dict(gene=r["gene"], pdb=r["pdb"], cohort=r["cohort"], level=LEVELS[lv] if lv is not None else None,
                         recall=f("recall_pair"), same_site=f("same_site"), z_pred=f("z_pred"), dockq_nr=f("DockQ_Nr")))
    L.append(f"\nCLIP motif atlas ({len(rows)} entries): best-chain interface level {_share([r['level'] for r in rows])}")
    for lv in LEVELS:
        d = [r for r in rows if r["level"] == lv]
        if not d:
            continue
        g = lambda k, t: 100 * np.mean(t(np.array([r[k] for r in d if np.isfinite(r[k])], float)))
        L.append(f"  {lv:10s} n={len(d):3d}: footprint recall >=0.5 {g('recall', lambda v: v >= 0.5):.0f}%, "
                 f"AF3 footprint on native site {g('same_site', lambda v: v > 0):.0f}%, "
                 f"median z_AF3 {np.nanmedian([r['z_pred'] for r in d]):.2f}, median DockQ_Nr {np.nanmedian([r['dockq_nr'] for r in d]):.2f}")
    return rows


def v2_main():
    logging.disable(logging.CRITICAL)
    L = []
    _write(v2_rnacompete_v4(L), "applications_v2_rnacompete_v4.tsv")
    _write(v2_rnacompete_v5(L), "applications_v2_rnacompete_v5.tsv")
    _write(v2_clip(L), "applications_v2_clip.tsv")
    print("\n".join(L))
    with open(os.path.join(OUTDIR, "applications_v2_summary.txt"), "w") as fh:
        fh.write("\n".join(L) + "\n")


# =========================================================================== #
# Decision tree v3 = v3 fold side (decision_tree.develop_v3) + v2 interface side
# =========================================================================== #
FOLD3 = fw.FOLD_LEVELS
RANK3 = {l: i for i, l in enumerate(FOLD3)}


def _share3(levels):
    c = Counter(l for l in levels if l)
    n = sum(c.values())
    return ", ".join(f"{l} {100 * c[l] / n:.0f}%" for l in FOLD3 if c[l]) + f" (n={n})" if n else "n=0"


def v3_rnacompete_v5(L):
    """v3 fold level of every probe chain (synthetic probes: not familiar; 2-4 RNA chains per complex)."""
    import json as _json
    import sqlite3
    import exp_probeMax_combined as ec
    from decision_tree.af3_output import read_prediction
    with contextlib.redirect_stdout(io.StringIO()):
        recs = ec.build_competition_records(ec.load_pred_rows(V5_DB), ec.load_rna_exp_map(V5_CSV),
                                            ec.load_domain_family_map(V5_DB))
    tpl = {}
    for eid, t in sqlite3.connect(V5_DB).execute("SELECT exp_db_id, RBP_PDB_Templates FROM pred_protein_rna"):
        try:
            v = _json.loads(t) if t else None
            tpl[str(eid).upper()] = bool(v) if isinstance(v, list) else None
        except _json.JSONDecodeError:
            tpl[str(eid).upper()] = None
    af = _first(os.path.join(DATA, "all_data", "other_analyses", "rnacompete", "rnacompete_v5_competition", "af"),
                os.path.join(DATA, "rnaCompete_v5_MINMAX", "protein_rna", "af"))   # fetch_data.sh --structures
    rows = []
    for r in recs:
        cif = os.path.join(af, f"fold_{r['pdb_id'].lower()}_s1_model_0.cif")
        if not os.path.exists(cif):
            continue
        pred = read_prediction(cif)
        by_chain = {c["chain"]: c for c in pred["rna_chains"]}
        for c in r["chains"]:
            pc = by_chain.get(c["chain"])
            c["fold3"] = (fw.fold_level_v3(rna_pae=pc["rna_pae"], n_rna=pred["n_rna"],
                                           rbp_template=tpl.get(r["pdb_id"].upper()), seq_familiar=False,
                                           tm_familiar=False) if pc else None)
            rows.append(dict(complex=r["pdb_id"], chain=c["chain"], role=c["role"], n_rna=pred["n_rna"],
                             rbp_template=tpl.get(r["pdb_id"].upper()), fold_level_v3=c["fold3"]))
        ranks = [RANK3[c["fold3"]] for c in r["chains"] if c.get("fold3") in RANK3]
        r["best3"] = max(ranks) if ranks else None
    mx = [x["fold_level_v3"] for x in rows if x["role"] == "maxprobe"]
    mn = [x["fold_level_v3"] for x in rows if x["role"] == "minprobe"]
    L.append(f"\nv5 RNAcompete competition complexes ({len(recs)}): v3 FOLD level, maxprobe chains {_share3(mx)}")
    L.append(f"  minprobe chains {_share3(mn)}; AUC(fold level, maxprobe) = "
             f"{_auc([RANK3[l] for l in mx if l in RANK3], [RANK3[l] for l in mn if l in RANK3]):.3f}")
    L.append("  note: every v5 complex holds 2-4 probe RNAs, so v3's '>= 2 RNA chains' point applies to all of them")
    metrics = (("rna_ipTM", "ipTM_r"), ("RNAmotif_score_rank", "motif score"))
    for lv, name in enumerate(FOLD3):
        sub = [r for r in recs if r.get("best3") == lv]
        if not sub:
            continue
        wr = ec.competition_win_rates(sub, tuple((k, lab, "") for k, lab in metrics))
        au = ec.competition_auc(sub, tuple((k, lab, "") for k, lab in metrics))
        L.append(f"  best chain {name:9s} n={len(sub):3d}: " + "; ".join(
            f"{lab} P(first) {wr[k]['rate']:.2f} AUC {au[k]['auc'] if au[k]['auc'] is not None else float('nan'):.3f}"
            for k, lab in metrics))
    return rows


V4_CONF = _first(os.path.join(DATA, "all_data", "other_analyses", "rnacompete", "rnacompete_v4_single_probe",
                              "rnacompete-seed-1_sample-0-confidence_files"),   # fetch_data.sh --structures
                 os.path.join(DATA, "rnaCompete_v4", "protein_rna", "rnacompete-seed-1_sample-0-confidence_files"))


def _v4_confidences():
    """exp_db_id -> (intra-RNA PAE of chain B, summary_confidences) from the AF3 seed-1 sample-0
    confidence JSONs (file stem 'rncmpt00001maxgauaauurbd_v3_002261' = id 'RNCMPT00001MAXGAUAAUURBDV3002261')."""
    import json as _json
    out = {}
    for f in sorted(os.listdir(V4_CONF)):
        if not f.endswith("_summary_confidences.json"):
            continue
        stem = f[:-len("_summary_confidences.json")]
        full = os.path.join(V4_CONF, stem + "_confidences.json")
        if not os.path.exists(full):
            continue
        d = _json.load(open(full))
        arr, ids = np.asarray(d["pae"], dtype=np.float32), np.asarray(d["token_chain_ids"])
        m = ids == "B"                                   # every v4 complex: protein A, probe RNA B
        out[stem.upper().replace("_", "")] = (float(arr[np.ix_(m, m)].mean()) if m.any() else None,
                                              _json.load(open(os.path.join(V4_CONF, f))))
    return out


def v3_rnacompete_v4(L):
    """v3 fold level of every single-probe complex (synthetic probes: not familiar; one RNA chain)."""
    import json as _json
    import sqlite3
    import probe_ranking_within_protein as pr
    conf = _v4_confidences()
    tpl, db_iptm = {}, {}
    for eid, t, ip in sqlite3.connect(V4_DB).execute(
            "SELECT exp_db_id, RBP_PDB_Templates, af3_rna_ipTM FROM pred_protein_rna"):
        try:
            v = _json.loads(t) if t else None
            tpl[str(eid).upper()] = bool(v) if isinstance(v, list) else None
        except _json.JSONDecodeError:
            tpl[str(eid).upper()] = None
        db_iptm[str(eid).upper()] = _scalar(ip)
    # consistency of the downloaded JSONs with the database (RNA chain ipTM)
    diff = [abs(c[1]["chain_iptm"][1] - db_iptm[k]) for k, c in conf.items()
            if db_iptm.get(k) is not None and len(c[1].get("chain_iptm", [])) > 1]
    proteins = pr.load_proteins(V4_DB)
    rows = []
    for p in proteins:
        for probe in p["maxprobes"] + p["minprobes"]:
            k = str(probe["pdb_id"]).upper()
            pae = conf.get(k, (None, None))[0]
            probe["fold3"] = (fw.fold_level_v3(rna_pae=pae, n_rna=1, rbp_template=tpl.get(k), seq_familiar=False,
                                               tm_familiar=False) if pae is not None else None)
            rows.append(dict(protein=p["protein"], probe=probe["pdb_id"], cls=probe["class"], rna_pae=pae,
                             rbp_template=tpl.get(k), fold_level_v3=probe["fold3"]))
        ranks = [RANK3[pp["fold3"]] for pp in p["maxprobes"] + p["minprobes"] if pp.get("fold3") in RANK3]
        p["best3"] = max(ranks) if ranks else None
    n_conf = sum(r["rna_pae"] is not None for r in rows)
    mx = [r["fold_level_v3"] for r in rows if r["cls"] == "maxprobe"]
    mn = [r["fold_level_v3"] for r in rows if r["cls"] == "minprobe"]
    paes = np.array([r["rna_pae"] for r in rows if r["rna_pae"] is not None])
    L.append(f"\nv4 RNAcompete single-probe complexes ({len(proteins)} RBPs, {len(rows)} probes; confidence JSONs for "
             f"{n_conf}; JSON vs DB RNA ipTM max |diff| {max(diff) if diff else float('nan'):.3f}): v3 FOLD level, "
             f"maxprobes {_share3(mx)}")
    L.append(f"  minprobes {_share3(mn)}; AUC(fold level, maxprobe) = "
             f"{_auc([RANK3[l] for l in mx if l in RANK3], [RANK3[l] for l in mn if l in RANK3]):.3f}")
    mxp = [r["rna_pae"] for r in rows if r["cls"] == "maxprobe" and r["rna_pae"] is not None]
    mnp = [r["rna_pae"] for r in rows if r["cls"] == "minprobe" and r["rna_pae"] is not None]
    L.append(f"  intra-RNA PAE median {np.median(paes):.1f} A (<=4 A {100 * np.mean(paes <= 4):.0f}%, 4-8 A "
             f"{100 * np.mean((paes > 4) & (paes <= 8)):.0f}%, >8 A {100 * np.mean(paes > 8):.0f}%); "
             f"AUC(-PAE_r, maxprobe) = {_auc([-x for x in mxp], [-x for x in mnp]):.3f}")
    L.append(f"  RBP template used: {100 * np.mean([bool(r['rbp_template']) for r in rows]):.0f}% of probes")
    for lv, name in enumerate(FOLD3):
        sub = [p for p in proteins if p.get("best3") == lv]
        if not sub:
            continue
        parts = []
        for key, lab in (("af3_rna_ipTM", "ipTM_r"), ("RNAmotif_score", "motif score")):
            acc = list(pr.rank_accuracy(sub, key).values())
            parts.append(f"{lab} per-protein AUC {np.mean(acc):.3f}" if acc else f"{lab} n/a")
        L.append(f"  RBPs with best probe {name:9s} n={len(sub):3d}: " + "; ".join(parts))
    return rows


def v3_clip(L):
    """v3 fold level of each CLIP-atlas entry (deposited structures: pre-cutoff entries are familiar)."""
    from decision_tree.develop_v3 import build_rows, post_inputs
    from decision_tree.validate import PP, _model_index
    models = _model_index()
    db, fam, _, _ = post_inputs()
    post, _ = build_rows(db, models, fam, lambda pid, lddt: 0.0, lambda *a: None)
    import sqlite3
    pre_ids = [str(r[0]).upper() for r in sqlite3.connect(PP["b2021.db"]).execute(
        "SELECT exp_db_id FROM pred_protein_rna")]
    pre, _ = build_rows(PP["b2021.db"], models, {p: (1.0, 1.0, 999) for p in pre_ids},
                        lambda pid, lddt: 0.0, lambda *a: None)
    lv = {}
    pae = {2: 3.0, 1: 6.0, 0: 9.0}
    for r in post + pre:
        if r["base_score"] is None:
            continue
        lv[r["pdb"]] = fw.fold_level_v3(rna_pae=pae[int(r["base_score"])], n_rna=r["n_rnas"],
                                         rbp_template=r["rbp_template"], seq_familiar=r["seq_familiar"],
                                         tm_familiar=r["tm_familiar"])
    pos = list(csv.DictReader(open(os.path.join(CLIP, "positional.tsv")), delimiter="\t"))
    rows = []
    for r in pos:
        f = lambda k: float(r[k]) if r.get(k) not in ("", "nan", None) else np.nan
        rows.append(dict(gene=r["gene"], pdb=r["pdb"], cohort=r["cohort"], fold_level_v3=lv.get(r["pdb"].upper()),
                         recall=f("recall_pair"), same_site=f("same_site"), z_pred=f("z_pred"), dockq_nr=f("DockQ_Nr")))
    L.append(f"\nCLIP motif atlas ({len(rows)} entries): v3 FOLD level {_share3([r['fold_level_v3'] for r in rows])}")
    for name in FOLD3:
        d = [r for r in rows if r["fold_level_v3"] == name]
        if not d:
            continue
        g = lambda k, t: 100 * np.mean(t(np.array([r[k] for r in d if np.isfinite(r[k])], float)))
        L.append(f"  {name:9s} n={len(d):3d}: footprint recall >=0.5 {g('recall', lambda v: v >= 0.5):.0f}%, "
                 f"AF3 footprint on native site {g('same_site', lambda v: v > 0):.0f}%, "
                 f"median z_AF3 {np.nanmedian([r['z_pred'] for r in d]):.2f}, "
                 f"median DockQ_Nr {np.nanmedian([r['dockq_nr'] for r in d]):.2f}")
    return rows


def v3_main():
    """Decision tree v3 on the binding-specificity sets: interface side = v2 (rerun), fold side = v3.
    RNAcompete v4 fold side: intra-RNA PAE from the downloaded AF3 confidence JSONs (V4_CONF)."""
    logging.disable(logging.CRITICAL)
    L = ["Decision tree v3 (fold side v3, interface side v2) on the binding-specificity sets", "",
         "INTERFACE SIDE (v2, unchanged)"]
    _write(v2_rnacompete_v4(L), "applications_v3_interface_rnacompete_v4.tsv")
    _write(v2_rnacompete_v5(L), "applications_v3_interface_rnacompete_v5.tsv")
    _write(v2_clip(L), "applications_v3_interface_clip.tsv")
    L += ["", "FOLD SIDE (v3)"]
    _write(v3_rnacompete_v4(L), "applications_v3_fold_rnacompete_v4.tsv")
    _write(v3_rnacompete_v5(L), "applications_v3_fold_rnacompete_v5.tsv")
    _write(v3_clip(L), "applications_v3_fold_clip.tsv")
    print("\n".join(L))
    with open(os.path.join(OUTDIR, "applications_v3_summary.txt"), "w") as fh:
        fh.write("\n".join(L) + "\n")


def _write(rows, name):
    if rows:
        with open(os.path.join(OUTDIR, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, list(rows[0].keys()), delimiter="\t")
            w.writeheader()
            w.writerows(rows)


# =========================================================================== #
# Probability form (decision_tree.probability): P_fold and P_interface levels
# =========================================================================== #
from decision_tree import probability as pb

PLV = pb.LEVELS
PRANK = {l: i for i, l in enumerate(PLV)}


def _share_p(levels):
    c = Counter(l for l in levels if l)
    n = sum(c.values())
    return ", ".join(f"{l} {100 * c[l] / n:.0f}%" for l in PLV if c[l]) + f" (n={n})" if n else "n=0"


def _lv(p):
    return pb.level(p) if p is not None else None


def _templates(db):
    """number of RNA-bound protein templates AF3 used per entry (None if unknown)"""
    import json as _json
    import sqlite3
    tpl = {}
    for eid, t in sqlite3.connect(db).execute("SELECT exp_db_id, RBP_PDB_Templates FROM pred_protein_rna"):
        try:
            v = _json.loads(t) if t else None
            tpl[str(eid).upper()] = len(v) if isinstance(v, list) else None
        except _json.JSONDecodeError:
            tpl[str(eid).upper()] = None
    return tpl


def _auc_lv(mx, mn):
    return _auc([PRANK[l] for l in mx if l], [PRANK[l] for l in mn if l])


def p_rnacompete_v4(L):
    """single-probe complexes: one probe RNA (chain B), synthetic (not familiar)."""
    import probe_ranking_within_protein as pr
    conf = _v4_confidences()
    tpl = _templates(V4_DB)
    proteins = pr.load_proteins(V4_DB)
    rows = []
    for p in proteins:
        for probe in p["maxprobes"] + p["minprobes"]:
            f, k = probe["features"], str(probe["pdb_id"]).upper()
            ptm = f["af3_protein_pTM"] if f["af3_protein_pTM"] == f["af3_protein_pTM"] else None
            pl = f["af3_rna_pLDDT_avg"]
            pl = None if pl is None or pl != pl else (pl / 100 if pl > 1 else pl)
            pae = conf.get(k, (None, None))[0]
            t = tpl.get(k)
            pf = pb.fold_probability(pae, 1, False, False, None if t is None else t > 0)[0]       # synthetic probes: not familiar
            pi = pb.interface_probability(f["af3_chain_pair_pae_min"], ptm, False, False, pl)[0]
            probe.update(pf=pf, pi=pi)
            rows.append(dict(protein=p["protein"], probe=probe["pdb_id"], cls=probe["class"], p_fold=pf,
                             fold_level=_lv(pf), p_interface=pi, interface_level=_lv(pi)))
    for side, key in (("FOLD", "fold_level"), ("INTERFACE", "interface_level")):
        mx = [r[key] for r in rows if r["cls"] == "maxprobe"]
        mn = [r[key] for r in rows if r["cls"] == "minprobe"]
        L.append(f"\nv4 RNAcompete single-probe complexes ({len(proteins)} RBPs, {len(rows)} probes): {side} "
                 f"maxprobes {_share_p(mx)}; minprobes {_share_p(mn)}; AUC(level, maxprobe) {_auc_lv(mx, mn):.3f}")
    return rows


def p_rnacompete_v5(L):
    """competition complexes: 2-4 synthetic probe RNAs per complex."""
    import sqlite3
    import exp_probeMax_combined as ec
    from decision_tree.af3_output import read_prediction
    with contextlib.redirect_stdout(io.StringIO()):
        recs = ec.build_competition_records(ec.load_pred_rows(V5_DB), ec.load_rna_exp_map(V5_CSV),
                                            ec.load_domain_family_map(V5_DB))
    tpl = _templates(V5_DB)
    meta = {eid: (ptm, pl) for eid, ptm, pl in sqlite3.connect(V5_DB).execute(
        "SELECT exp_db_id, af3_protein_pTM, af3_rna_pLDDT_avg FROM pred_protein_rna")}
    af = _first(os.path.join(DATA, "all_data", "other_analyses", "rnacompete", "rnacompete_v5_competition", "af"),
                os.path.join(DATA, "rnaCompete_v5_MINMAX", "protein_rna", "af"))   # fetch_data.sh --structures
    rows = []
    for r in recs:
        cif = os.path.join(af, f"fold_{r['pdb_id'].lower()}_s1_model_0.cif")
        pred = read_prediction(cif) if os.path.exists(cif) else None
        by_chain = {c["chain"]: c for c in pred["rna_chains"]} if pred else {}
        longest = max(pred["rna_chains"], key=lambda c: c["length"]) if pred and pred["rna_chains"] else None
        ptm, pl = (_scalar(x) for x in meta.get(r["pdb_id"], (None, None)))
        pl = None if pl is None else (pl / 100 if pl > 1 else pl)
        t = tpl.get(r["pdb_id"].upper())
        pf = (pb.fold_probability(longest.get("rna_pae"), pred["n_rna"], False, False, None if t is None else t > 0)[0]
              if longest else None)
        for c in r["chains"]:
            pc = by_chain.get(c["chain"])
            pae = -c["pae_oriented"] if np.isfinite(c.get("pae_oriented", np.nan)) else None
            cpl = pc["rna_plddt"] if pc and pc.get("rna_plddt") is not None else pl
            pi = pb.interface_probability(pae, ptm, False, False, cpl)[0]
            c.update(pi=pi)
            rows.append(dict(complex=r["pdb_id"], chain=c["chain"], role=c["role"], p_fold=pf, fold_level=_lv(pf),
                             p_interface=pi, interface_level=_lv(pi)))
        ps = [c["pi"] for c in r["chains"] if c.get("pi") is not None]
        r["best_i"] = PRANK[_lv(max(ps))] if ps else None
        r["fold"] = PRANK[_lv(pf)] if pf is not None else None
    for side, key in (("FOLD (longest probe)", "fold_level"), ("INTERFACE", "interface_level")):
        mx = [x[key] for x in rows if x["role"] == "maxprobe"]
        mn = [x[key] for x in rows if x["role"] == "minprobe"]
        L.append(f"\nv5 RNAcompete competition complexes ({len(recs)}): {side} maxprobe chains {_share_p(mx)}; "
                 f"minprobe chains {_share_p(mn)}")
    metrics = (("rna_ipTM", "ipTM_r"), ("RNAmotif_score_rank", "motif score"))
    for lv, name in enumerate(PLV):
        sub = [r for r in recs if r.get("best_i") == lv]
        if not sub:
            continue
        wr = ec.competition_win_rates(sub, tuple((k, lab, "") for k, lab in metrics))
        au = ec.competition_auc(sub, tuple((k, lab, "") for k, lab in metrics))
        L.append(f"  best chain interface {name:9s} n={len(sub):3d}: " + "; ".join(
            f"{lab} P(first) {wr[k]['rate']:.2f} AUC {au[k]['auc'] if au[k]['auc'] is not None else float('nan'):.3f}"
            for k, lab in metrics))
    return rows


def p_clip(L):
    """CLIP-atlas entries (deposited structures; pre-cutoff entries are familiar): P_fold of the longest RNA chain,
    best P_interface over the RNA chains (AF3 chain-pair PAE minimum to the contacting protein)."""
    import json as _json
    import sqlite3
    from decision_tree.develop import familiarity
    from decision_tree.develop_v3 import build_rows, post_inputs
    from decision_tree.develop_probability import af3_values, _af3_cache
    from decision_tree.validate import PP, _model_index
    models = _model_index()
    db, fam, _, _ = post_inputs()
    post, _ = build_rows(db, models, fam, lambda pid, lddt: 0.0, lambda *a: None)
    pre_ids = [str(r[0]).upper() for r in sqlite3.connect(PP["b2021.db"]).execute(
        "SELECT exp_db_id FROM pred_protein_rna")]
    pre, _ = build_rows(PP["b2021.db"], models, {p: (1.0, 1.0, 999) for p in pre_ids},
                        lambda pid, lddt: 0.0, lambda *a: None)
    cif = {}
    for name in ("b2021.db", "a2021.db"):
        for e, f in sqlite3.connect(PP[name]).execute("SELECT exp_db_id, FileName FROM pred_protein_rna"):
            cif[str(e).strip().upper()] = models.get(os.path.basename(str(f)).lower())
    from decision_tree.af3_output import read_prediction
    from decision_tree.develop_probability import PDB_SEQ, PRED_TM
    tm_post = {p: v[0] for p, v in _json.load(open(PRED_TM)).get("post", {}).items()}
    seq_post = _json.load(open(PDB_SEQ)).get("post", {})
    pf, pi = {}, {}
    for r, period in [(r, "post") for r in post] + [(r, "pre") for r in pre]:
        k = r["pdb"].upper()
        if not cif.get(k):
            continue
        pred = read_prediction(cif[k])
        if not pred["rna_chains"]:
            continue
        # pre-cutoff entries are themselves in the pre-cutoff library, so their fold is familiar
        fam = True if period == "pre" else (tm_post.get(k) is not None and tm_post[k] >= 0.5)
        seq = True if period == "pre" else (seq_post.get(k) or 0) >= 0.8
        longest = max(pred["rna_chains"], key=lambda c: c["length"])
        pf[k] = pb.fold_probability(longest.get("rna_pae"), pred["n_rna"], fam, seq, r.get("rbp_template"))[0]
        ps = [pb.interface_probability(c["pae_min"], c["partner_ptm"], fam, seq, c["rna_plddt"])[0] for c in pred["rna_chains"]]
        ps = [p for p in ps if p is not None]
        if ps:
            pi[k] = max(ps)
    pos = list(csv.DictReader(open(os.path.join(CLIP, "positional.tsv")), delimiter="\t"))
    rows = []
    for r in pos:
        k = r["pdb"].upper()
        f = lambda c: float(r[c]) if r.get(c) not in ("", "nan", None) else np.nan
        rows.append(dict(gene=r["gene"], pdb=r["pdb"], cohort=r["cohort"], p_fold=pf.get(k), fold_level=_lv(pf.get(k)),
                         p_interface=pi.get(k), interface_level=_lv(pi.get(k)),
                         recall=f("recall_pair"), same_site=f("same_site"), z_pred=f("z_pred"), dockq_nr=f("DockQ_Nr")))
    for side, key in (("FOLD", "fold_level"), ("INTERFACE (best chain)", "interface_level")):
        L.append(f"\nCLIP motif atlas ({len(rows)} entries): {side} {_share_p([r[key] for r in rows])}")
        for name in PLV:
            d = [r for r in rows if r[key] == name]
            if not d:
                continue
            g = lambda c, t: 100 * np.mean(t(np.array([r[c] for r in d if np.isfinite(r[c])], float)))
            L.append(f"  {name:9s} n={len(d):3d}: footprint recall >=0.5 {g('recall', lambda v: v >= 0.5):.0f}%, "
                     f"AF3 footprint on native site {g('same_site', lambda v: v > 0):.0f}%, "
                     f"median z_AF3 {np.nanmedian([r['z_pred'] for r in d]):.2f}, "
                     f"median DockQ_Nr {np.nanmedian([r['dockq_nr'] for r in d]):.2f}")
    return rows


def p_main():
    """Probability form on the binding-specificity sets (paper)."""
    logging.disable(logging.CRITICAL)
    L = ["Calibrated probabilities (decision_tree.probability) on the binding-specificity sets",
         "levels: WEAK < 25% <= MODERATE < 60% <= STRONG"]
    _write(p_rnacompete_v4(L), "applications_probability_rnacompete_v4.tsv")
    _write(p_rnacompete_v5(L), "applications_probability_rnacompete_v5.tsv")
    _write(p_clip(L), "applications_probability_clip.tsv")
    print("\n".join(L))
    with open(os.path.join(OUTDIR, "applications_probability_summary.txt"), "w") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    p_main() if "--probability" in sys.argv else (v3_main() if "--v3" in sys.argv else v2_main())
