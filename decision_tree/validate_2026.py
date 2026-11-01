"""
Independent test of the decision trees on PDB entries released in 2026 (data/all_data_zip/all_databases/2026/protein_rna.db).

The 2026 targets are used STRICTLY as a test set: every tree is applied exactly as released
(tree.py thresholds and node orders, calibration.json design and levels); nothing is re-fitted,
re-selected or re-calibrated on them, and they share no PDB entry with the pre-/post-cutoff sets.

Compared, per side:
  FOLD       AF3 confidence only (mean intra-RNA PAE level, the v2 measure)
             gated tree as first drawn (16b, tree.ORDER_DRAWN)
             gated tree confidence first (16c, tree.ORDER)
             scored framework v2 (17; confidence + familiar RNA)
  INTERFACE  interface tier only (ipTM_r / chain-pair PAE)
             gated trees 16b and 16c
             scored framework v2 (tier + familiar RNA - warning sign)

Outcomes, defined exactly as in the paper:
  fold       LDDT_r > 0.85 of the (longest) RNA chain: backbone C3' LDDT_r (fold_label_c3, the paper) or, with
             DT_LDDT=allatom, the all-atom LDDT_r (RNA_LDDT_allAtom; decision_tree/lddt_labels.py).
  interface  per RNA chain: correct if its best protein-RNA DockQ (RBP_DockQ) >= 0.23, no protein-RNA
             DockQ = incorrect (reproduces the a2021 placement labels for 585/586 chains with a DockQ;
             all 58 a2021 chains without one are incorrect); complexes with no RBP_DockQ are skipped.

Familiar RNA (v2): as in decision_tree.develop, against the PRE-cutoff structures only: >= 80% RNA
sequence identity to a pre-cutoff protein-RNA RNA, or US-align TM-score >= 0.5 (longest RNA chain
>= 20 nt) to any pre-cutoff RNA chain.
Gated-tree answers come from the database columns the pipeline fills, as in validate_tree:
RNAFamily, RBP_PDB_Templates (mainExecution.py --addMSAtemp), RNA_Pseudoknots (+dssr) and the InterPro
domains (--addDomains). MSA depth is not used: AlphaFold Server downloads contain no RNA alignment
(checked with 7QR3, RNA MSA 26 sequences in the local AF3 run) and their protein alignments come from
other databases than the paper's local runs (7QR3: 10,263 vs 7,152 sequences). The gated trees are
therefore evaluated without their "deep RNA MSA" question (its exit leaf removed, the order of the
other leaves unchanged). Unknown answers are passed as unknown (the tree then returns a range,
scored at its midpoint, as for any user without that information).

Primary criterion, fixed before looking at the results: AUC on the 2026 targets; differences with a
paired bootstrap over complexes (2,000 resamples). Secondary: agreement of the observed level rates
with the post-cutoff calibration (Wilson 95% CIs).

Run from the repo root (RNABench env):  python -m decision_tree.validate_2026
Outputs: results/plots/all_chains/decision_tree/validation_2026/
"""

from __future__ import annotations

import contextlib
import csv
import glob
import io
import json
import os
import sqlite3
import subprocess
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from decision_tree.af3_output import read_prediction
from decision_tree.develop import SEQ_FAMILIAR, TM_FAMILIAR, wilson
from decision_tree.framework import calibration, fold_confidence, warnings
from decision_tree.rules import interface_tier
from decision_tree.tree import (FOLD_LEAVES, INTERFACE_LEAVES, ORDER, ORDER_DRAWN, fold_answers,
                                interface_answers)
from decision_tree.validate import OUTDIR as DT_OUT, PP, _auc
from decision_tree.validate_tree import _family, _first, _j, _pk, _tpl, leaf_rank

from evaluation.bundle_paths import PROTEIN_RNA_DIR, Y2026_DB as DB26
DATA26 = PROTEIN_RNA_DIR["2026"]          # published structures (pdb/, af/, lddt/) of the 2026 test set
OUTDIR = os.path.join(DT_OUT, "validation_2026")
SIM = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "structural_similarity")
PRE_CHAINS = os.path.join(SIM, "chains", "pre")
USALIGN = os.path.join(_REPO_ROOT, "evaluation", "USalign", "USalign")
GOOD_LDDT, ACCEPTABLE = 0.85, 0.23
N_BOOT = 2000
RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------- #
# file indices
# --------------------------------------------------------------------------- #
def _index(pattern, key):
    out = {}
    for p in sorted(glob.glob(os.path.join(DATA26, "**", pattern), recursive=True)):
        out.setdefault(key(p), p)
    return out


MODELS = _index("fold_*_s1_model_0.cif", lambda p: os.path.basename(p).split("_")[1].upper())
ALIGNED = _index("*_pdb_aligned_rna.pdb", lambda p: os.path.basename(p)[:4].upper())
EXP_CIFS = {}
for _p in sorted(glob.glob(os.path.join(DATA26, "pdb", "*.cif"))):
    EXP_CIFS.setdefault(os.path.basename(_p)[:-4].upper(), _p)


# --------------------------------------------------------------------------- #
# outcomes
# --------------------------------------------------------------------------- #
def _longest_rna(path):
    import gemmi
    st = gemmi.read_structure(path)
    st.setup_entities()
    best = None
    for ch in st[0]:
        pol = ch.get_polymer()
        if len(pol) and pol.check_polymer_type() == gemmi.PolymerType.Rna and (best is None or len(pol) > len(best[1])):
            best = (ch.name, pol)
    return best


def _write_chain(chain_pol, out):
    import gemmi
    name, pol = chain_pol
    s, m, c = gemmi.Structure(), gemmi.Model("1"), gemmi.Chain(name[:1])
    for r in pol:
        c.add_residue(r)
    m.add_chain(c)
    s.add_model(m)
    s.setup_entities()
    s.write_pdb(out)


_FOLD_LDDT = None


def fold_label(pid):
    """LDDT_r of the (longest) RNA chain under the label of decision_tree/lddt_labels.py: backbone C3' (default, the
    paper; fold_label_c3) or all-atom (RNA_LDDT_allAtom of the 2026 database)"""
    global _FOLD_LDDT
    if _FOLD_LDDT is None:
        _FOLD_LDDT = {}
    if pid not in _FOLD_LDDT:
        from decision_tree.lddt_labels import LABEL, complex_lddt
        _FOLD_LDDT[pid] = fold_label_c3(pid) if LABEL == "backbone" else complex_lddt(DB26, "pred_protein_rna").get(pid)
    return _FOLD_LDDT[pid]


def fold_label_c3(pid):
    """backbone C3' LDDT_r of the longest RNA chain, the label used until October 2026. From the aligned RNA files
    of the pipeline; where these are missing, from the structures directly (superposition-free, so the same
    value: reproduces 95/95 labels, |difference| <= 0.003)."""
    import tempfile
    from evaluation.extractEvalMetrics import calculate_complex_lddt
    ref = ALIGNED.get(pid)
    with contextlib.redirect_stdout(io.StringIO()):
        if ref is not None:
            return calculate_complex_lddt(ref, ref.replace("_pdb_aligned_rna", "_af_aligned_rna"))
        if pid not in EXP_CIFS or pid not in MODELS:
            return None
        a, b = _longest_rna(EXP_CIFS[pid]), _longest_rna(MODELS[pid])
        if not a or not b:
            return None
        with tempfile.TemporaryDirectory() as d:
            _write_chain(a, os.path.join(d, "r.pdb"))
            _write_chain(b, os.path.join(d, "m.pdb"))
            return calculate_complex_lddt(os.path.join(d, "r.pdb"), os.path.join(d, "m.pdb"))


from evaluation.bundle_paths import ost_binary
OST = ost_binary()


def local_dockq(pid):
    """Per-interface DockQ from a local OpenStructure comparison (same settings as the SWISS-MODEL
    step: --dockq --min-nuc-length 3), for complexes without RBP_DockQ; reproduces 9ZIK exactly."""
    import subprocess
    out = os.path.join(OUTDIR, "ost", f"{pid.lower()}.json")
    if not os.path.exists(out):
        if pid not in EXP_CIFS or pid not in MODELS:
            return None
        os.makedirs(os.path.dirname(out), exist_ok=True)
        subprocess.run([OST, "compare-structures", "-m", MODELS[pid], "-r", EXP_CIFS[pid], "--dockq",
                        "--min-nuc-length", "3", "-ft", "-o", out], capture_output=True, text=True)
    try:
        d = json.load(open(out))
    except (OSError, json.JSONDecodeError):
        return None
    return [{f"{i[2]}:{i[3]}": v} for i, v in zip(d.get("dockq_interfaces") or [], d.get("dockq") or [])]


def interface_labels(dockq_raw, protein_chains, rna_chains):
    dq = _j(dockq_raw)
    if not isinstance(dq, list):
        return None
    d = {k: v for x in dq if isinstance(x, dict) for k, v in x.items() if v is not None}
    out = {}
    for c in rna_chains:
        vals = [v for k, v in d.items() if c in k.split(":") and any(p in k.split(":") for p in protein_chains)]
        if vals:      # an RNA chain without any protein-RNA DockQ has no reliable label and is left out
            out[c] = max(vals) >= ACCEPTABLE
    return out


# --------------------------------------------------------------------------- #
# RNA familiarity against the pre-cutoff structures
# --------------------------------------------------------------------------- #
def sequence_identity(targets):
    from evaluation.postpre_matched_similarity import candidate_pairs, load
    pre = load(PP["b2021.db"], "pred_protein_rna", False)
    post = [t for t in load(DB26, "pred_protein_rna", False) if t["id"] in targets]
    with contextlib.redirect_stdout(io.StringIO()):
        cands = candidate_pairs(post, pre, False)
    return {pid: max((r for _, r, _ in rows), default=0.0) for pid, rows in cands.items()}


def _pre_chain_list():
    import gemmi
    if not os.path.isdir(PRE_CHAINS) or not any(f.endswith(".pdb") for f in os.listdir(PRE_CHAINS)):
        # one PDB file per pre-cutoff RNA chain, rebuilt from the structures (as for the paper)
        from evaluation.structural_similarity_to_pre import build_chain_sets
        print("  pre-cutoff RNA chain library missing: rebuilding it from the pre-cutoff structures ...", flush=True)
        build_chain_sets()
    valid = set()
    from evaluation.postpre_matched_similarity import SYSTEMS, load
    for cfg in SYSTEMS.values():
        valid |= {r["id"][:4].lower() for r in load(cfg["pre"], cfg["table"], cfg["protein"])}
    out = []
    for f in sorted(os.listdir(PRE_CHAINS)):
        if f.endswith(".pdb") and f[:4] in valid:
            out.append((f, gemmi.read_structure(os.path.join(PRE_CHAINS, f))[0][0].get_polymer().length()))
    return out


def structural_tm(targets):
    from evaluation.structural_similarity_to_pre import MIN_LEN_RATIO, MIN_NT_TM, rna_chains
    work = os.path.join(OUTDIR, "chains")
    os.makedirs(work, exist_ok=True)
    pre = _pre_chain_list()
    out = {}
    for pid in sorted(targets):
        cif = EXP_CIFS.get(pid)
        if cif is None:
            continue
        chains = rna_chains(cif, work, pid.lower())
        if not chains:
            continue
        name, _, n = max(chains, key=lambda t: t[2])
        if n < MIN_NT_TM:
            out[pid] = (None, n)
            continue
        cands = [f for f, ln in pre if ln >= MIN_LEN_RATIO * n]
        lst = os.path.join(work, name + ".list")
        open(lst, "w").write("\n".join(cands) + "\n")
        res = subprocess.run([USALIGN, os.path.join(work, name), "-dir2", PRE_CHAINS + "/", lst,
                              "-mol", "RNA", "-outfmt", "2"], capture_output=True, text=True)
        tms = [float(l.split("\t")[2]) for l in res.stdout.splitlines() if l.strip() and not l.startswith("#")]
        out[pid] = (max(tms) if tms else 0.0, n)
    return out


def _no_msa(order, leaves=FOLD_LEAVES):
    """Gated fold tree without its "deep RNA MSA" question: drop the node and its exit leaf."""
    i = order.index("msa")
    return tuple(o for o in order if o != "msa"), tuple(l for k, l in enumerate(leaves) if k != i)


# --------------------------------------------------------------------------- #
# rows
# --------------------------------------------------------------------------- #
def collect():
    con = sqlite3.connect(DB26)
    q = ("SELECT p.exp_db_id, p.FileName, p.ProteinChainIDs, p.RNAChainIDs, p.af3_rna_pLDDT_avg, "
         "p.af3_global_pae_avg, p.RBP_DockQ, e.RNAFamily, p.RNA_Pseudoknots, p.RBP_PDB_Templates, e.domain_names FROM pred_protein_rna p JOIN exp_protein_rna e "
         "ON upper(trim(p.exp_db_id)) = upper(trim(e.PDBId))")
    recs = con.execute(q).fetchall()
    ids = {str(r[0]).strip().upper() for r in recs}

    # leakage guard: no 2026 entry may be part of the pre-/post-cutoff sets the trees were built on
    used = set()
    for db in ("a2021.db", "b2021.db"):
        used |= {str(x[0]).strip().upper() for x in sqlite3.connect(PP[db]).execute(
            "SELECT exp_db_id FROM pred_protein_rna")}
    overlap = ids & used
    assert not overlap, f"2026 entries also in the development sets: {sorted(overlap)}"

    cache = os.path.join(OUTDIR, "familiarity_cache.json")
    fam = json.load(open(cache)) if os.path.exists(cache) else {}
    if set(fam.get("ids", [])) != ids:
        print(f"2026 complexes: {len(ids)}; familiarity: sequence identity ...", flush=True)
        seq = sequence_identity(ids)
        print("  US-align against the pre-cutoff RNA chains ...", flush=True)
        tm = structural_tm(ids)
        os.makedirs(OUTDIR, exist_ok=True)
        json.dump(dict(ids=sorted(ids), seq=seq, tm=tm), open(cache, "w"))
    else:
        seq, tm = fam["seq"], {k: tuple(v) for k, v in fam["tm"].items()}
        todo = {p for p in ids if p not in tm and p in EXP_CIFS}
        if todo:
            print(f"  US-align for {len(todo)} targets without a structural comparison ...", flush=True)
            tm.update(structural_tm(todo))
            json.dump(dict(ids=sorted(ids), seq=seq, tm=tm), open(cache, "w"))

    fold, iface = [], []
    for eid, fn, pch, rch, pl, gpae, dq, fam, pk, tpl_raw, doms in recs:
        pid = str(eid).strip().upper()
        cif = MODELS.get(pid)
        if cif is None:
            continue
        pred = read_prediction(cif)
        if not pred["rna_chains"]:
            continue
        s = seq.get(pid)
        t, ln = tm.get(pid, (None, None))
        familiar = (s is not None and s >= SEQ_FAMILIAR) or (t is not None and ln >= 20 and t >= TM_FAMILIAR)
        tpl = _tpl(tpl_raw)
        longest = max(pred["rna_chains"], key=lambda c: c["length"])
        base = dict(pdb=pid, familiar=familiar, seq_identity=s, tm_pre=t, rbp_template=tpl)

        lddt = fold_label(pid)
        if lddt is not None:
            conf = fold_confidence(rna_pae=longest["rna_pae"])
            a = fold_answers(conserved_family=_family(fam), pseudoknot=_pk(pk),
                             rna_plddt=_first(pl), global_pae=_first(gpae))
            a.pop("msa")
            fold.append(dict(base, rna_lddt=lddt, outcome=lddt > GOOD_LDDT, rna_pae=longest["rna_pae"],
                             conf=conf, v2=None if conf is None else conf + int(familiar),
                             g16b=leaf_rank(a, *_no_msa(ORDER_DRAWN["fold"])),
                             g16c=leaf_rank(a, *_no_msa(ORDER["fold"])), **{f"q_{k}": v for k, v in a.items()}))

        pc = _j(pch, [])
        pc = pc if isinstance(pc, list) else [pc]
        if not isinstance(_j(dq), list):              # no SWISS-MODEL DockQ: local OpenStructure
            dq = local_dockq(pid)
        labels = interface_labels(dq, [str(x) for x in pc], [c["chain"] for c in pred["rna_chains"]])
        if labels is None:
            continue
        for c in pred["rna_chains"]:
            tier = interface_tier(c["rna_iptm"], c["pae_min"])
            if tier is None or c["chain"] not in labels:
                continue
            ts = {"bad": 0, "mid": 1, "good": 2}[tier]
            warn = warnings(partner_ptm=c["partner_ptm"], rna_plddt=c["rna_plddt"])
            a = interface_answers(protein_ptms=[c["partner_ptm"]] if c["partner_ptm"] is not None else None,
                                  rna_plddt=c["rna_plddt"], rbp_template=tpl,
                                  domains=[d for d in str(doms or "").split(";") if d.strip() not in ("", "-")] or None,
                                  rna_iptm=c["rna_iptm"], pair_pae=c["pae_min"])
            iface.append(dict(base, chain=c["chain"], outcome=labels[c["chain"]], tier=ts,
                              v2=ts - len(warn) + int(familiar), warning=";".join(warn),
                              g16b=leaf_rank(a, ORDER_DRAWN["interface"], INTERFACE_LEAVES),
                              g16c=leaf_rank(a, ORDER["interface"], INTERFACE_LEAVES),
                              **{f"q_{k}": v for k, v in a.items()}))
    return fold, iface


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def auc(rows, key):
    d = [r for r in rows if r[key] is not None]
    return _auc([r[key] for r in d if r["outcome"]], [r[key] for r in d if not r["outcome"]]), len(d)


def boot_diff(rows, a, b):
    """Paired bootstrap over complexes of AUC(a) - AUC(b)."""
    rows = [r for r in rows if r[a] is not None and r[b] is not None]
    by = {}
    for r in rows:
        by.setdefault(r["pdb"], []).append(r)
    keys = list(by)
    diffs = []
    for _ in range(N_BOOT):
        s = [r for k in RNG.choice(keys, len(keys)) for r in by[k]]
        if len({r["outcome"] for r in s}) < 2:
            continue
        diffs.append(auc(s, a)[0] - auc(s, b)[0])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return auc(rows, a)[0] - auc(rows, b)[0], lo, hi


def level_of(score):
    return 0 if score <= 1 else (1 if score == 2 else 2)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    fold, iface = collect()
    cal = calibration()
    L = [f"Independent 2026 test set ({DB26}); trees applied as released, nothing refitted.",
         f"fold: {len(fold)} protein-bound RNAs (longest RNA chain), {sum(r['outcome'] for r in fold)} well folded; "
         f"interface: {len(iface)} RNA chains in {len({r['pdb'] for r in iface})} complexes, "
         f"{sum(r['outcome'] for r in iface)} correct",
         f"familiar RNA: {sum(r['familiar'] for r in fold)}/{len(fold)} targets "
         f"(seq >= {SEQ_FAMILIAR:.0%}: {sum((r['seq_identity'] or 0) >= SEQ_FAMILIAR for r in fold)}, "
         f"TM >= {TM_FAMILIAR}: {sum((r['tm_pre'] or 0) >= TM_FAMILIAR for r in fold)})",
         ""]
    # reference: the same MSA-free gated fold trees on the pre-/post-cutoff protein-RNA sets
    from decision_tree.validate_tree import auc_of, fold_rows
    ref = [dict(r) for r in fold_rows() if r["system"] == "protein-RNA"]
    for r in ref:
        r.pop("msa", None)
    L.append("reference, protein-RNA fold AUC of the MSA-free gated trees: " + "; ".join(
        f"{per} 16b {auc_of([r for r in ref if r['period'] == per], *_no_msa(ORDER_DRAWN['fold'])):.2f} / "
        f"16c {auc_of([r for r in ref if r['period'] == per], *_no_msa(ORDER['fold'])):.2f}" for per in ("pre", "post")))
    L.append("")
    sides = (("FOLD (well folded, LDDT_r > 0.85)", fold,
              [("conf", "AF3 confidence only (intra-RNA PAE)"), ("g16b", "gated tree, as first drawn (16b)"),
               ("g16c", "gated tree, confidence first (16c)"), ("v2", "scored framework v2 (17)")],
              cal.get("fold", {}).get("protein-RNA", {}).get("post", {}), ("WEAK", "MODERATE", "STRONG")),
             ("INTERFACE (correct, DockQ >= 0.23)", iface,
              [("tier", "interface tier only"), ("g16b", "gated tree, as first drawn (16b)"),
               ("g16c", "gated tree, confidence first (16c)"), ("v2", "scored framework v2 (17)")],
              cal.get("interface", {}).get("post", {}), ("UNRELIABLE", "CAUTION", "RELIABLE")))
    summary = {}
    for title, rows, cands, cal_post, names in sides:
        L.append(title)
        res = {k: auc(rows, k) for k, _ in cands}
        for k, name in cands:
            L.append(f"  {name:42s} AUC {res[k][0]:.3f} (n={res[k][1]})")
        best = max(res, key=lambda k: res[k][0])
        L.append(f"  highest AUC: {dict(cands)[best]}")
        for k, name in cands:
            if k == best:
                continue
            d, lo, hi = boot_diff(rows, best, k)
            L.append(f"    vs {name:39s} dAUC {d:+.3f} [95% CI {lo:+.3f}, {hi:+.3f}]")
        L.append("  v2 levels on 2026 vs post-cutoff calibration (% outcome [95% CI], n):")
        for lv, nm in enumerate(names):
            s = [r["outcome"] for r in rows if r["v2"] is not None and level_of(r["v2"]) == lv]
            p, lo, hi = wilson(sum(s), len(s))
            c = cal_post.get(nm, {})
            L.append(f"    {nm:11s} 2026 {100 * p:5.1f}% [{100 * lo:.0f}-{100 * hi:.0f}] n={len(s):3d} | "
                     f"post-cutoff {100 * c.get('rate', np.nan):.0f}% [{100 * c.get('ci_lo', np.nan):.0f}-"
                     f"{100 * c.get('ci_hi', np.nan):.0f}] n={c.get('n', 0)}")
        summary[title.split()[0].lower()] = dict(auc={k: res[k][0] for k in res}, n={k: res[k][1] for k in res},
                                                 best=best)
        L.append("")
    for name, rows in (("fold.tsv", fold), ("interface.tsv", iface)):
        with open(os.path.join(OUTDIR, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
            w.writeheader()
            w.writerows(rows)
    json.dump(summary, open(os.path.join(OUTDIR, "summary.json"), "w"), indent=1, default=float)
    open(os.path.join(OUTDIR, "summary.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
