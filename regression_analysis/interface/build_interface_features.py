"""
Feature table for the interface-failure feature-group analysis (one row per protein-RNA interface).

Outcome: the placement class of each native protein-RNA interface (evaluation/interface_rna_pose.py;
'same place, same orientation' merged into the correct-site class, as in the paper).

Features, by group (prefix = group key; complex-level unless stated):
  af3   AF3 confidence: RNA / protein pLDDT, global PAE, minimum chain-pair PAE, RNA / protein (i)pTM,
        ranking score
  exp   experimental and environmental: method, resolution, pH, temperature, listed ions and
        ligands, ionic strength, PEG, buffer; Mg2+ and monovalent-salt concentration and HEPES / Tris
        buffer of the deposited solution recipe (concentrations as parsed, mM; missing when no recipe
        is deposited); numbers of ions (total and Mg, Zn, Mn, K, Na) and ligands in the deposited
        structure
  seq   RNA sequence and structure of the target: total / longest RNA length, GC content, sequence
        complexity, entropy, MFE, ensemble diversity, centroid energy, and the target's share of
        syn nucleotides, sugar puckers other than C3'-endo and gamma angles other than g+ (from the
        experimental structure, never from the AF3 model)
  evo   evolutionary: RNA and protein MSA depth, Rfam family annotated; for post-cutoff targets also
        the sequence identity and fold similarity (US-align TM) of the closest pre-cutoff RNA
  tmpl  templates: numbers of RBP (protein-RNA) and protein-only templates used by AF3
  comp  composition: numbers of protein and RNA chains, total protein length, protein / RNA length
        ratio, identical protein copies, identical RNA copies
Nothing that defines the placement classes (DockQ, contact match, pocket RMSD, pose) and no
accuracy of the AF3 model enters the features.

Writes regression_analysis/interface/results/interface_features_{post,pre}.tsv.
Run in the RNABench env:  python regression_analysis/interface/build_interface_features.py
"""

import csv
import json
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (REPO,):
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluation.interface_class_rna_geometry import complex_features

OUT = os.path.join(HERE, "results")
POSTPRE = os.path.join(REPO, "data", "compareDatabases", "post_pre", "a2021_b2021_protein_rna")
DBS = {"post": os.path.join(POSTPRE, "a2021.db"), "pre": os.path.join(POSTPRE, "b2021.db")}
REG_CSV = {"post": "a2021", "pre": "b2021"}
REG_DIR = os.path.join(REPO, "regression_analysis", "csvs4analysis_with_interface")
POSE_TSV = [os.path.join(REPO, "results", d, "all_chains", "interface_rna_pose", "interface_rna_pose.tsv")
            for d in ("plots", "plots_odl")]
MERGE = {"same place, same orientation": "right protein site, wrong RNA pose"}

# regression-table columns per group (complex level)
REG_COLS = {
    "af3": ["af3_rna_pLDDT_avg", "af3_protein_pLDDT_avg", "af3_global_pae_avg", "af3_chain_pair_pae_min",
            "af3_rna_ipTM_mean", "af3_protein_ipTM_mean", "af3_protein_pTM_mean", "af3_ranking_score"],
    "exp": ["is_xray", "is_em", "XRayResolution", "exp_pH", "exp_Temp", "ionic_strength", "peg_percent",
            "buffer_m", "has_peg"],
    "seq": ["rna_total_length", "rna_max_chain_length", "agg_rna_gc_content", "agg_rna_sequence_complexity",
            "agg_rna_shannon_entropy_k2", "RNA_MFE_Value", "RNA_Ensemble_Diversity", "RNA_Centroid_Energy"],
    "evo": ["log10_rna_msa_size", "log10_protein_msa_size"],
    "tmpl": ["n_rbp_templates", "n_protein_templates"],
    "comp": ["NumberProteins", "NumberRNAs", "protein_total_length", "protein_to_rna_length_ratio"],
}


def _list(raw):
    if raw is None:
        return []
    t = str(raw).strip()
    if t.startswith("["):
        try:
            return json.loads(t)
        except ValueError:
            return []
    return [t] if t else []


def _num(v):
    try:
        v = float(v)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def labels():
    pose = next(p for p in POSE_TSV if os.path.isfile(p))
    d = pd.read_csv(pose, sep="\t")
    d = d[d.dockq.notna()].copy()
    d["placement"] = d["placement"].replace(MERGE)
    d["pdb"] = d["pdb"].str.upper()
    return d[["period", "pdb", "native_pair", "model_pair", "placement"]]


def regression_features(period):
    frames = []
    for name in ("rna_protein_1v1", "multichain_complex"):
        p = os.path.join(REG_DIR, REG_CSV[period], f"{REG_CSV[period]}__{name}.csv")
        frames.append(pd.read_csv(p, low_memory=False))
    df = pd.concat(frames, ignore_index=True)
    df["pdb"] = df["exp_db_id"].astype(str).str.strip().str.upper()
    df = df.drop_duplicates("pdb")
    out = pd.DataFrame({"pdb": df["pdb"]})
    for g, cols in REG_COLS.items():
        for c in cols:
            out[f"{g}__{c}"] = pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
    return out


def db_features(period):
    """Target conformations, Rfam family, identical copies (from the period database)."""
    con = sqlite3.connect(DBS[period])
    exp = {str(r[0]).upper(): r[1] for r in con.execute("SELECT PDBId, RNAFamily FROM exp_protein_rna")}
    pred = {str(r[0]).upper(): r[1:] for r in con.execute(
        "SELECT exp_db_id, ProteinSequence, RNASequence FROM pred_protein_rna")}
    con.close()
    geo = complex_features(DBS[period])
    rows = []
    for pid, (pseq, rseq) in pred.items():
        ps, rs = [str(s) for s in _list(pseq)], [str(s) for s in _list(rseq)]
        fam = [f for f in _list(exp.get(pid)) if str(f) not in ("None", "", "nan")]
        g = geo.get(pid, {})
        rows.append({"pdb": pid,
                     "seq__target_syn_share": g.get("exp_syn", np.nan),
                     "seq__target_non_c3endo_share": g.get("exp_non_c3endo", np.nan),
                     "seq__target_non_gplus_share": g.get("exp_non_gplus", np.nan),
                     "evo__rfam_family": float(bool(fam)),
                     "comp__identical_protein_copies": float(len(set(ps)) < len(ps)),
                     "comp__identical_rna_copies": float(len(set(rs)) < len(rs))})
    return pd.DataFrame(rows)


def _ion_counts(raw):
    out = {}
    try:
        items = json.loads(str(raw).replace("'", '"').replace("True", "true").replace("False", "false"))
    except ValueError:
        return out
    for it in items if isinstance(items, list) else []:
        if isinstance(it, dict):
            out[str(it.get("ions", "")).upper()] = out.get(str(it.get("ions", "")).upper(), 0) + _num(it.get("n", 0))
    return out


def exp_extra(period):
    """Solution recipe (Mg2+, monovalent salt, HEPES / Tris) and deposited ions per species."""
    frames = [pd.read_csv(os.path.join(REG_DIR, REG_CSV[period], f"{REG_CSV[period]}__{n}.csv"), low_memory=False)
              for n in ("rna_protein_1v1", "multichain_complex")]
    df = pd.concat(frames, ignore_index=True)
    df["pdb"] = df["exp_db_id"].astype(str).str.strip().str.upper()
    df = df.drop_duplicates("pdb")
    sol = [c for c in df.columns if c.startswith("sol_conc_")]
    mg = [c for c in sol if ("mg" in c or "magnes" in c) and "mg_ml" not in c]
    mono = [c for c in sol if any(k in c for k in ("nacl", "kcl", "sodium_chloride", "potassium_chloride",
                                                    "lithium_chloride", "_k_acetate", "na_acetate",
                                                    "sodium_acetate", "ammonium_acetate"))]
    num = lambda cols: df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(1) if cols else 0.0
    out = pd.DataFrame({"pdb": df["pdb"],
                        "exp__solution_mg_mm": num(mg), "exp__solution_monovalent_salt_mm": num(mono),
                        "exp__buffer_hepes": (num([c for c in sol if "hepes" in c]) > 0).astype(float),
                        "exp__buffer_tris": (num([c for c in sol if "tris" in c and "bis_tris" not in c]) > 0).astype(float)})
    con = sqlite3.connect(DBS[period])
    rows = con.execute("SELECT PDBId, Ions, Ligands, exp_Solution FROM exp_protein_rna").fetchall()
    con.close()
    ions = {str(r[0]).upper(): _ion_counts(r[1]) for r in rows}
    ligs = {str(r[0]).upper(): sum(_ion_counts(str(r[2]).replace("'ligands'", "'ions'")).values()) for r in rows}
    recipe = {str(r[0]).upper() for r in rows if str(r[3] or "").strip() not in ("", "None", "nan", "[]", "{}")}
    no_recipe = ~out["pdb"].isin(recipe)
    for c in ("exp__solution_mg_mm", "exp__solution_monovalent_salt_mm", "exp__buffer_hepes", "exp__buffer_tris"):
        out.loc[no_recipe, c] = np.nan                        # no recipe deposited: unknown, not zero
    out["exp__deposited_ions_total"] = out["pdb"].map(lambda p: sum(ions.get(p, {}).values()))
    out["exp__deposited_ligands_total"] = out["pdb"].map(lambda p: ligs.get(p, 0.0))
    for sp in ("MG", "ZN", "MN", "K", "NA"):
        out[f"exp__deposited_{sp.lower()}_ions"] = out["pdb"].map(lambda p, sp=sp: ions.get(p, {}).get(sp, 0.0))
    return out


def familiarity():
    """Post-cutoff targets: identity / TM of the closest pre-cutoff RNA (decision-tree inputs)."""
    from decision_tree.develop_v3 import post_inputs
    _db, fam, _f, _i = post_inputs()
    return pd.DataFrame([{"pdb": k, "evo__seq_identity_to_pre": _num(v[0]), "evo__tm_to_pre": _num(v[1])}
                         for k, v in fam.items()])


def build(period):
    lab = labels()
    lab = lab[lab.period == period].copy()
    df = lab.merge(regression_features(period), on="pdb", how="left")
    df = df.merge(db_features(period), on="pdb", how="left")
    df = df.merge(exp_extra(period), on="pdb", how="left")
    if period == "post":
        df = df.merge(familiarity(), on="pdb", how="left")
    df["n_interfaces_of_complex"] = df.groupby("pdb")["pdb"].transform("size")
    return df


def main():
    os.makedirs(OUT, exist_ok=True)
    for period in ("post", "pre"):
        df = build(period)
        feats = [c for c in df.columns if "__" in c]
        cov = df[feats].notna().mean().sort_values()
        df.to_csv(os.path.join(OUT, f"interface_features_{period}.tsv"), sep="\t", index=False)
        print(f"{period}: {len(df)} interfaces, {df.pdb.nunique()} complexes, {len(feats)} features")
        print("  lowest coverage:", ", ".join(f"{k} {v:.0%}" for k, v in cov.head(6).items()))
        print("  classes:", df.placement.value_counts().to_dict())


if __name__ == "__main__":
    main()
