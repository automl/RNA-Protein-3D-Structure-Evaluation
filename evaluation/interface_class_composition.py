"""
Composition of the interface placement classes (SI table).

For every native protein-RNA interface of the pre-/post-cutoff sets (placement classes of
evaluation/interface_rna_pose.py), this joins:
  - the global-fold class (i-iv) of its complex, with the paper's rule
    (evaluationMetrics.get_sample_audit_global_fold_interface_lists):
      (i)   RNA_LDDT > 0.85 and Protein_LDDT > 0.85 and Protein_TM > 0.85
            (longest RNA >= 60 nt: also RNA_TM > 0.5)
      (ii)  RNA_LDDT > 0.85, otherwise
      (iii) RNA_LDDT <= 0.85 and Protein_LDDT > 0.85 and Protein_TM > 0.85
      (iv)  RNA_LDDT <= 0.85, otherwise;
  - the per-chain fold accuracy of the two chains of the interface: TM and LDDT of its protein chain
    (Protein_TM_AllChains / Protein_LDDT_AllChains) and of its RNA chain (RNA_TM_AllChains /
    RNA_LDDT_AllChains), so a correctly folded chain in the wrong position can be told apart from a
    misfolded one;
  - the placement of the protein in the assembly: the lowest DockQ of the protein-protein interfaces
    of the interface's protein chain (OpenStructure compare-structures JSON of the complex; complexes
    with one protein chain have none);
  - composition: numbers of protein and RNA chains, identical protein copies.

Writes to results/plots/all_chains/interface_class_composition/:
  interface_class_composition.tsv          one row per interface
  interface_class_composition_summary.tsv  one row per period x placement class
  interface_class_composition_table.tex    SI table (post-cutoff; pre-cutoff once its per-chain
                                           protein columns are filled)
Run in the RNABench env:  python evaluation/interface_class_composition.py
"""

import glob
import json
import os
import sqlite3

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from evaluation.bundle_paths import PROTEIN_RNA_DB, PROTEIN_RNA_DIR
PERIODS = dict(PROTEIN_RNA_DB)
# Processing folders whose lddt/ holds the full-complex OpenStructure comparison JSONs
OST_FOLDERS = [PROTEIN_RNA_DIR["post"], PROTEIN_RNA_DIR["pre"]]
POSE_TSV = [os.path.join(_REPO_ROOT, "results", d, "all_chains", "interface_rna_pose", "interface_rna_pose.tsv")
            for d in ("plots", "plots_odl")]
OUTDIR = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "interface_class_composition")

# Placement classes as shown in the paper (Fig. interface_failure_modes B)
CLASSES = [("correct interface", "Correct interface"),
           ("same place, duplex slid", "Correct site, slid duplex"),
           ("right protein site, wrong RNA pose", "Correct site, incorrect RNA"),
           ("same place, RNA turned", "Correct site, rotated RNA"),
           ("shifted nearby", "Shifted site"),
           ("different region", "Distal site"),
           ("protein binding region mispredicted", "Incorrect binding site")]
MERGE = {"same place, same orientation": "right protein site, wrong RNA pose"}
FOLD_CLASSES = ["i", "ii", "iii", "iv"]
DOCKQ_OK = 0.23


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def _list(raw):
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return list(raw)
    txt = str(raw).strip()
    if not txt or txt in ("None", "null"):
        return []
    if txt.startswith("["):
        try:
            return json.loads(txt)
        except ValueError:
            return []
    return [txt]


def _float(v):
    try:
        v = float(v)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _keyed(raw):
    """[{"A": 0.9}, {"B": 0.8}] -> {"A": 0.9, "B": 0.8}."""
    out = {}
    for item in _list(raw):
        if isinstance(item, dict):
            for k, v in item.items():
                out[str(k)] = _float(v)
    return out


def _per_chain(raw, chain_ids, chain_map=None):
    """Per-predicted-chain scores from an *_AllChains column -> ({chain: value}, {pred: ref index}).
    Current format: [{"B": 0.47}, ...]. Older databases store the all-vs-all matrix
    (rows = reference chains, columns = predicted chains in RNAChainIDs order), reduced here with the
    pipeline's one-to-one assignment; LDDT is read for the reference chain assigned by TM (chain_map)."""
    items = _list(raw)
    if not items or not isinstance(items[0], (list, tuple)):
        return _keyed(raw), None
    from evaluation.chain_assignment import assigned_by_pred_chain
    refs = [(i,) for i in range(len(items))]
    preds = [(c,) for c in chain_ids][:len(items[0])]
    if chain_map is not None:
        out = {}
        for j, (c,) in enumerate(preds):
            i = chain_map.get(c)
            if i is not None and i < len(items) and j < len(items[i]):
                v = _float(items[i][j])
                if np.isfinite(v) and v > 0:
                    out[c] = v
        return out, chain_map
    values, cmap = assigned_by_pred_chain(items, refs, preds, positive_only=True)
    return {k: _float(v) for d in values for k, v in d.items()}, cmap


def fold_class(rna_lddt, protein_lddt, protein_tm, rna_tm, rna_len_longest):
    """Global-fold class i-iv of a complex (paper rule); None if RNA_LDDT or Protein_LDDT is missing."""
    if not (np.isfinite(rna_lddt) and np.isfinite(protein_lddt)):
        return None
    protein_ok = protein_lddt > 0.85 and np.isfinite(protein_tm) and protein_tm > 0.85
    if rna_lddt > 0.85:
        long_rna_ok = not (np.isfinite(rna_len_longest) and rna_len_longest >= 60.0) or (
            np.isfinite(rna_tm) and rna_tm > 0.5)
        return "i" if protein_ok and long_rna_ok else "ii"
    return "iii" if protein_ok else "iv"


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def complexes(period, db):
    """exp_db_id -> per-complex values from the period database."""
    con = sqlite3.connect(db)
    cols = {r[1] for r in con.execute("PRAGMA table_info(pred_protein_rna)")}
    want = ["exp_db_id", "ProteinChainIDs", "RNAChainIDs", "ProteinSequence", "RNALength",
            "RNA_LDDT", "Protein_LDDT", "RNA_TM", "Protein_TM",
            "RNA_TM_AllChains", "RNA_LDDT_AllChains", "Protein_TM_AllChains", "Protein_LDDT_AllChains"]
    sel = [c if c in cols else f"NULL AS {c}" for c in want]
    rows = con.execute(f"SELECT {', '.join(sel)} FROM pred_protein_rna").fetchall()
    con.close()
    out = {}
    for r in rows:
        d = dict(zip(want, r))
        pid = str(d["exp_db_id"]).strip().upper()
        lens = [_float(x) for x in _list(d["RNALength"])]
        seqs = [str(s) for s in _list(d["ProteinSequence"])]
        d.update(
            pcs=[str(c) for c in _list(d["ProteinChainIDs"])],
            rcs=[str(c) for c in _list(d["RNAChainIDs"])],
            identical_protein_copies=len(set(seqs)) < len(seqs),
            fold_class=fold_class(_float(d["RNA_LDDT"]), _float(d["Protein_LDDT"]), _float(d["Protein_TM"]),
                                  _float(d["RNA_TM"]), max(lens) if lens else np.nan),
            rna_tm_chain=None, rna_lddt_chain=None,
            prot_tm_chain=_keyed(d["Protein_TM_AllChains"]), prot_lddt_chain=_keyed(d["Protein_LDDT_AllChains"]),
        )
        rna_ids = [str(c) for c in _list(d["RNAChainIDs"])]
        d["rna_tm_chain"], cmap = _per_chain(d["RNA_TM_AllChains"], rna_ids)
        d["rna_lddt_chain"], _ = _per_chain(d["RNA_LDDT_AllChains"], rna_ids, cmap)
        out[pid] = d
    return out


def ost_index():
    idx = {}
    for folder in OST_FOLDERS:
        for p in glob.glob(os.path.join(folder, "lddt", "fold_*_s1_model_0_lddt.json")):
            idx.setdefault(os.path.basename(p).split("_")[1].upper(), p)
    return idx


def protein_protein_dockq(json_path, protein_chains):
    """[(model chain 1, model chain 2, DockQ)] of the protein-protein interfaces of a complex."""
    if not json_path:
        return []
    try:
        d = json.load(open(json_path)) or {}
    except (OSError, ValueError):
        return []
    ints, dq = d.get("dockq_interfaces") or [], d.get("dockq") or []
    return [(it[2], it[3], _float(q)) for it, q in zip(ints, dq)
            if len(it) == 4 and it[2] in protein_chains and it[3] in protein_chains]


# --------------------------------------------------------------------------- #
def build():
    pose = next((p for p in POSE_TSV if os.path.isfile(p)), None)
    if pose is None:
        raise SystemExit("interface_rna_pose.tsv not found; run evaluation/interface_rna_pose.py first")
    d = pd.read_csv(pose, sep="\t")
    d = d[d["dockq"].notna()].copy()
    d["placement"] = d["placement"].replace(MERGE)
    meta = {period: complexes(period, db) for period, db in PERIODS.items()}
    idx = ost_index()
    pp_cache = {}
    rows = []
    for r in d.itertuples(index=False):
        c = meta.get(r.period, {}).get(str(r.pdb).upper())
        if c is None:
            continue
        mp, mr = str(r.model_pair).split(":")       # model-file chain names (OST JSON)
        dp, dr = str(r.pred_pair).split(":")        # DB chain names (*_AllChains keys)
        key = (r.period, r.pdb)
        if key not in pp_cache:
            pp_cache[key] = protein_protein_dockq(idx.get(str(r.pdb).upper()), set(c["pcs"]) | {mp})
        partner = [q for a, b, q in pp_cache[key] if mp in (a, b) and np.isfinite(q)]
        rows.append(dict(
            period=r.period, pdb=r.pdb, native_pair=r.native_pair, pred_pair=r.pred_pair,
            placement=r.placement, dockq=r.dockq, fold_class=c["fold_class"],
            n_protein=len(c["pcs"]), n_rna=len(c["rcs"]), identical_protein_copies=c["identical_protein_copies"],
            protein_chain_tm=c["prot_tm_chain"].get(dp, np.nan),
            protein_chain_lddt=c["prot_lddt_chain"].get(dp, np.nan),
            rna_chain_tm=c["rna_tm_chain"].get(dr, np.nan),
            rna_chain_lddt=c["rna_lddt_chain"].get(dr, np.nan),
            protein_rmsd_all=r.protein_rmsd_all, pocket_rmsd=r.pocket_rmsd,
            rna_translation=r.rna_translation, rna_rotation=r.rna_rotation,
            partner_pp_dockq_min=min(partner) if partner else np.nan,
            n_partner_pp=len(partner),
        ))
    return pd.DataFrame(rows)


def _med_iqr(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return np.nan, np.nan, np.nan, 0
    return s.median(), s.quantile(0.25), s.quantile(0.75), len(s)


def summarise(df):
    out = []
    for period in ("post", "pre"):
        for key, label in CLASSES:
            g = df[(df.period == period) & (df.placement == key)]
            if g.empty:
                continue
            row = dict(period=period, placement=label, n=len(g))
            fc = g.fold_class.dropna()
            for k in FOLD_CLASSES:
                row[f"class_{k}_pct"] = 100 * (fc == k).mean() if len(fc) else np.nan
            for col in ("protein_chain_tm", "protein_chain_lddt", "rna_chain_tm", "rna_chain_lddt",
                        "protein_rmsd_all", "pocket_rmsd"):
                m, q1, q3, n = _med_iqr(g[col])
                row.update({f"{col}_median": m, f"{col}_q1": q1, f"{col}_q3": q3, f"{col}_n": n})
            pp = g[g.n_partner_pp > 0]
            row["with_pp_partner_n"] = len(pp)
            row["pp_partner_wrong_pct"] = 100 * (pp.partner_pp_dockq_min < DOCKQ_OK).mean() if len(pp) else np.nan
            row["single_protein_pct"] = 100 * (g.n_protein == 1).mean()
            row["multi_rna_pct"] = 100 * (g.n_rna >= 2).mean()
            row["identical_protein_copies_pct"] = 100 * g.identical_protein_copies.mean()
            out.append(row)
    return pd.DataFrame(out)


def latex(summary, periods=("post", "pre")):
    """SI table: fold class, per-chain accuracy (protein TM and LDDT, RNA LDDT; TM-score is not
    informative for the short RNAs of most interfaces), protein-protein placement and composition,
    one block per period."""
    f = lambda v: "--" if not np.isfinite(v) else f"{v:.2f}"
    pct = lambda v: "--" if not np.isfinite(v) else f"{v:.0f}"
    name = {"post": "Post-cutoff", "pre": "Pre-cutoff"}
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Composition of the interface placement classes (post- and pre-cutoff protein-RNA "
        r"interfaces). Fold class of the complex (i, RNA and protein accurate; ii, RNA accurate; iii, protein "
        r"accurate; iv, neither); median TM-score and LDDT of the interface's protein chain and LDDT of its RNA "
        r"chain; share of interfaces whose protein chain is misplaced against its protein partners (lowest "
        r"protein-protein DockQ $<0.23$; complexes with several protein chains); share with two or more RNA "
        r"chains and with identical protein copies.}",
        r"\label{tab:placement_composition}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lrrrrrrrrrrr}", r"\toprule",
        r" & & \multicolumn{4}{c}{Fold class (\%)} & \multicolumn{2}{c}{Protein chain} & RNA chain & "
        r"Partner PP & $\geq2$ RNA & Identical \\",
        r"Placement & $n$ & i & ii & iii & iv & TM & LDDT & LDDT & wrong (\%) & chains (\%) & proteins (\%) \\",
    ]
    for period in periods:
        s = summary[summary.period == period]
        if s.empty:
            continue
        lines += [r"\midrule", r"\multicolumn{12}{l}{\emph{" + name[period] + r"}} \\"]
        for r in s.itertuples(index=False):
            lines.append(" & ".join([
                r.placement, str(r.n), pct(r.class_i_pct), pct(r.class_ii_pct), pct(r.class_iii_pct),
                pct(r.class_iv_pct), f(r.protein_chain_tm_median), f(r.protein_chain_lddt_median),
                f(r.rna_chain_lddt_median), pct(r.pp_partner_wrong_pct), pct(r.multi_rna_pct),
                pct(r.identical_protein_copies_pct)]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    df = build()
    df.to_csv(os.path.join(OUTDIR, "interface_class_composition.tsv"), sep="\t", index=False)
    summary = summarise(df)
    summary.to_csv(os.path.join(OUTDIR, "interface_class_composition_summary.tsv"), sep="\t", index=False)
    with open(os.path.join(OUTDIR, "interface_class_composition_table.tex"), "w") as fh:
        fh.write(latex(summary))
    cols = ["period", "placement", "n", "class_i_pct", "class_ii_pct", "class_iii_pct", "class_iv_pct",
            "protein_chain_tm_median", "protein_chain_lddt_median", "rna_chain_tm_median", "rna_chain_lddt_median",
            "pp_partner_wrong_pct", "multi_rna_pct", "identical_protein_copies_pct"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summary[cols].round(2).to_string(index=False))
    print(f"\n{len(df)} interfaces -> {OUTDIR}")


if __name__ == "__main__":
    main()
