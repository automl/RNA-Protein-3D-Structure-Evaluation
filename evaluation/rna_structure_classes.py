"""
Structural classes of the target RNAs and AF3 RNA accuracy per class, pre vs post cutoff.

Every target is assigned ONE structural class from its EXPERIMENTAL structure (exp_* tables;
using the prediction would be circular). Classes are tested in order of structural complexity
and the first match wins:

  L_shaped           tRNA (Rfam family) or an unlabelled 70-95 nt RNA with a multiway junction
                     and a 3'-CCA end (tRNA-like L-shaped fold)
  pseudoknot         >= 1 pseudoknot
  multiway_junction  >= 1 multibranch (>= 3-way) loop
  internal_loop      >= 1 internal loop or bulge
  hairpin            >= 1 hairpin loop
  duplex             paired, but no loop (intermolecular duplex)
  single_stranded    no base pairs

Element counts come from the RNA_Stems / RNA_HairpinLoops / RNA_InternalLoops /
RNA_MultibranchLoops / RNA_Pseudoknots columns (DSSR all-chain dot-bracket parsed with forgi;
each column is a list of nucleotide-position lists, one per element). When that annotation
failed (every list empty, including RNA_DanglingEnds) the class is taken directly from the
experimental DSSR JSON (stems, hairpins, iloops+bulges, n-way junctions, pseudoknot brackets).
Where both exist, the two class calls agree for >= 98% of targets.

Datasets: post-cutoff (a2021) and pre-cutoff (b2021), protein-RNA and RNA-only, read from
data/compareDatabases/post_pre/. Accuracy = RNA_LDDT (the same column that defines the
good-RNA classes i+ii, RNA_LDDT > 0.85) and RNA_TM.

stdlib only, so it runs in any environment. Usage:
  python evaluation/rna_structure_classes.py            # summary table + TSV
"""

import ast
import csv
import glob
import json
import math
import os
import re
import sqlite3
import statistics as st
from collections import Counter, defaultdict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_REPO_ROOT, "data")
_POSTPRE = os.path.join(_DATA, "compareDatabases", "post_pre")

CLASS_ORDER = ["L_shaped", "pseudoknot", "multiway_junction", "internal_loop",
               "hairpin", "duplex", "single_stranded"]
CLASS_LABELS = {"L_shaped": "L-shaped (tRNA)", "pseudoknot": "Pseudoknot",
                "multiway_junction": "Multiway junction", "internal_loop": "Internal loop",
                "hairpin": "Hairpin", "duplex": "Duplex", "single_stranded": "Single-stranded"}
GOOD_RNA_LDDT = 0.85      # classes i+ii (evaluationMetrics.get_sample_audit_global_fold_interface_lists)

# (system, period) -> sqlite DB, pred/exp tables, experimental-DSSR folders
DATASETS = {
    ("protein_rna", "post"): dict(
        db=os.path.join(_POSTPRE, "a2021_b2021_protein_rna", "a2021.db"),
        pred="pred_protein_rna", exp="exp_protein_rna",
        dssr=["all_data/a2021_post_cutoff/protein_rna/dssr",          # published layout (fetch_data.sh --structures)
             "a2021/a2021x/protein_rna/dssr", "a2021/a2021_nonX/protein_rna/dssr"]),
    ("protein_rna", "pre"): dict(
        db=os.path.join(_POSTPRE, "a2021_b2021_protein_rna", "b2021.db"),
        pred="pred_protein_rna", exp="exp_protein_rna",
        dssr=["all_data/b2021_pre_cutoff/protein_rna/dssr", "b2021x/protein_rna/dssr", "b2021_nonX/protein_rna/dssr"]),
    ("rna_only", "post"): dict(
        db=os.path.join(_POSTPRE, "a2021_b2021_rna_only", "a2021_rna.db"),
        pred="pred_rna_rna", exp="exp_rna_rna",
        dssr=["all_data/a2021_post_cutoff/rna_singlechain/dssr", "RNAonly/a2021_rnaOnly/rna_rna/dssr"]),
    ("rna_only", "pre"): dict(
        db=os.path.join(_POSTPRE, "a2021_b2021_rna_only", "b2021_rna.db"),
        pred="pred_rna_rna", exp="exp_rna_rna",
        dssr=["all_data/b2021_pre_cutoff/rna_singlechain/dssr", "RNAonly/b2021_rnaOnly/rna_rna/dssr"]),
}


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def _parse(raw):
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return raw
    s = str(raw).strip()
    if not s:
        return None
    for fn in (json.loads, ast.literal_eval):
        try:
            return fn(s)
        except Exception:
            pass
    return s


def _count(x):
    """Number of elements in a (possibly per-chain-nested) list of position lists."""
    if not isinstance(x, (list, tuple)) or not x:
        return 0
    if all(isinstance(i, (int, tuple)) for i in x):
        return 1
    return sum(_count(i) for i in x)


def _flat_ints(x):
    if isinstance(x, int):
        return [x]
    if isinstance(x, (list, tuple)):
        return [i for e in x for i in _flat_ints(e)]
    return []


def _as_list(raw):
    v = _parse(raw)
    if v is None:
        return []
    return [str(i) for i in v] if isinstance(v, list) else [str(v)]


def _float(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _clean_seq(s):
    return re.sub(r"\([^)]*\)", "N", s.upper()).replace("T", "U")


# --------------------------------------------------------------------------- #
# Secondary-structure element counts: DB columns or experimental DSSR
# --------------------------------------------------------------------------- #
def ss_from_db(stems, hairpins, iloops, multibranch, pknots, dangling):
    ss = {"Stems": _count(_parse(stems)), "HairpinLoops": _count(_parse(hairpins)),
          "InternalLoops": _count(_parse(iloops)), "MultibranchLoops": _count(_parse(multibranch)),
          "Pseudoknots": _count(_parse(pknots))}
    # A forgi failure leaves every list empty, including dangling ends; a genuinely
    # unpaired RNA records its whole span as a dangling end.
    ss["annotated"] = ss["Stems"] > 0 or len(_flat_ints(_parse(dangling))) > 0
    return ss


def ss_from_dssr(d):
    sstr = d.get("dbn", {}).get("all_chains", {}).get("sstr", "")
    nway = [int(m.group(1)) for j in d.get("junctions", []) or []
            for m in [re.match(r"(\d+)-way", j.get("type", ""))] if m]
    return {"Stems": len(d.get("stems", []) or []),
            "HairpinLoops": len(d.get("hairpins", []) or []),
            "InternalLoops": len(d.get("iloops", []) or []) + len(d.get("bulges", []) or []),
            "MultibranchLoops": sum(1 for n in nway if n >= 3),
            "Pseudoknots": int(any(c in sstr for c in "[{<")),
            "annotated": True}


def dssr_index(folders):
    idx = {}
    for d in folders:
        # *_pdb_dssr.json sorts after *_pdb_aligned_rna_dssr.json -> preferred when both exist
        for f in sorted(glob.glob(os.path.join(_DATA, d, "*_pdb*dssr.json")), reverse=True):
            idx.setdefault(os.path.basename(f).split("_")[0].upper(), f)
    return idx


def classify(ss, seqs, families, longest):
    if not ss["annotated"]:
        return "unannotated"
    known = [f for f in families if f and f.lower() != "none"]
    trna_like = (not known and 70 <= (longest or 0) <= 95 and ss["MultibranchLoops"] >= 1
                 and any(_clean_seq(s).endswith("CCA") for s in seqs))
    if (known and all(f == "tRNA" for f in known)) or trna_like:
        return "L_shaped"
    if ss["Pseudoknots"] >= 1:
        return "pseudoknot"
    if ss["MultibranchLoops"] >= 1:
        return "multiway_junction"
    if ss["InternalLoops"] >= 1:
        return "internal_loop"
    if ss["HairpinLoops"] >= 1:
        return "hairpin"
    if ss["Stems"] >= 1:
        return "duplex"
    return "single_stranded"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
CACHE_DIRS = [os.path.join(_REPO_ROOT, "results", d, "all_chains", "rna_structure_classes") for d in ("plots", "plots_odl")]


def load_classified(system, period):
    """One record per target: id, class, source ('db'/'dssr'), RNA_TM, RNA_LDDT, length, families.
    Also returns a dict of QC counts (DSSR coverage, rescued rows, DB-vs-DSSR agreement).
    The experimental DSSR files come with the structures (fetch_data.sh --structures). Without them, the records
    of a run with the structures are read from results/plots_odl (analysis_inputs.zip), so that the classes do not
    fall back to the database columns alone."""
    cfg = DATASETS[(system, period)]
    idx = dssr_index(cfg["dssr"])
    name = f"classified_{system}_{period}.json"
    if not idx:
        cached = next((os.path.join(d, name) for d in CACHE_DIRS if os.path.exists(os.path.join(d, name))), None)
        if cached:
            d = json.load(open(cached))
            return d["records"], Counter(d["qc"])
    recs, qc = _classify(cfg, idx)
    if idx:
        os.makedirs(CACHE_DIRS[0], exist_ok=True)
        json.dump(dict(records=recs, qc=dict(qc)), open(os.path.join(CACHE_DIRS[0], name), "w"))
    return recs, qc


def _classify(cfg, idx):
    con = sqlite3.connect(cfg["db"])
    rows = con.execute(f"""
        SELECT p.exp_db_id, p.RNA_TM, p.RNA_LDDT, e.RNASequence, e.RNAFamily, e.RNALength,
               e.RNA_Stems, e.RNA_HairpinLoops, e.RNA_InternalLoops, e.RNA_MultibranchLoops,
               e.RNA_Pseudoknots, e.RNA_DanglingEnds
        FROM {cfg['pred']} p JOIN {cfg['exp']} e
          ON UPPER(TRIM(p.exp_db_id)) = UPPER(TRIM(e.PDBId))""").fetchall()
    con.close()

    recs, qc = [], Counter()
    for pid, tm, lddt, seq, fam, ln, s, h, il, mb, pk, de in rows:
        seqs, fams = _as_list(seq), _as_list(fam)
        lens = [x for x in (_float(v) for v in _as_list(ln)) if x is not None]
        longest = max(lens) if lens else None
        cls = classify(ss_from_db(s, h, il, mb, pk, de), seqs, fams, longest)
        source = "db"
        path = idx.get(str(pid).strip().upper()[:4])
        if path:
            qc["dssr"] += 1
            try:
                cls_dssr = classify(ss_from_dssr(json.load(open(path))), seqs, fams, longest)
            except Exception:
                cls_dssr = None
            if cls_dssr:
                if cls == "unannotated":
                    cls, source = cls_dssr, "dssr"
                    qc["rescued"] += 1
                else:
                    qc["agree" if cls == cls_dssr else "disagree"] += 1
        qc[cls == "unannotated" and "unannotated" or "classified"] += 1
        recs.append(dict(pdb_id=str(pid).strip().upper(), cls=cls, source=source,
                         tm=_float(tm), lddt=_float(lddt), length=longest,
                         families=[f for f in fams if f and f.lower() != "none"]))
    qc["n"] = len(rows)
    return recs, qc


# --------------------------------------------------------------------------- #
# Statistics (stdlib)
# --------------------------------------------------------------------------- #
def _gammq(a, x):
    """Regularized upper incomplete gamma Q(a, x) (Numerical Recipes)."""
    if x <= 0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1:
        ap, s, d = a, 1.0 / a, 1.0 / a
        for _ in range(1000):
            ap += 1; d *= x / ap; s += d
            if abs(d) < abs(s) * 1e-12:
                break
        return 1.0 - s * math.exp(-x + a * math.log(x) - gln)
    b = x + 1 - a; c = 1e300; d = 1 / b; h = d
    for i in range(1, 1000):
        an = -i * (i - a); b += 2
        d = an * d + b; d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c; c = 1e-300 if abs(c) < 1e-300 else c
        d = 1 / d; de = d * c; h *= de
        if abs(de - 1) < 1e-12:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def kruskal(groups, min_n=5):
    """Kruskal-Wallis H (tie-corrected), df, p across groups with >= min_n values."""
    groups = [g for g in groups if len(g) >= min_n]
    if len(groups) < 2:
        return None
    allv = sorted((v, gi) for gi, g in enumerate(groups) for v in g)
    n = len(allv); ranks = [0.0] * n; tie = 0.0; i = 0
    while i < n:
        j = i
        while j + 1 < n and allv[j + 1][0] == allv[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        t = j - i + 1; tie += t ** 3 - t; i = j + 1
    rsum = defaultdict(float)
    for (_, gi), r in zip(allv, ranks):
        rsum[gi] += r
    h = 12 / (n * (n + 1)) * sum(rsum[gi] ** 2 / len(g) for gi, g in enumerate(groups)) - 3 * (n + 1)
    h /= (1 - tie / (n ** 3 - n))
    return h, len(groups) - 1, _gammq((len(groups) - 1) / 2, h / 2)


def summarize(recs):
    """{class: dict(n, median length/TM/LDDT, % good RNA)} for one dataset."""
    by = defaultdict(list)
    for r in recs:
        by[r["cls"]].append(r)
    out = {}
    for c in CLASS_ORDER:
        rs = by.get(c, [])
        lddt = [r["lddt"] for r in rs if r["lddt"] is not None]
        tm = [r["tm"] for r in rs if r["tm"] is not None]
        ln = [r["length"] for r in rs if r["length"] is not None]
        out[c] = dict(n=len(rs),
                      length=st.median(ln) if ln else float("nan"),
                      tm=st.median(tm) if tm else float("nan"),
                      lddt=st.median(lddt) if lddt else float("nan"),
                      pct_good=100 * sum(v > GOOD_RNA_LDDT for v in lddt) / len(lddt) if lddt else float("nan"),
                      families=Counter(f for r in rs for f in r["families"]))
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    outdir = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "publication_figures")
    os.makedirs(outdir, exist_ok=True)
    tsv_rows = []
    summ = {}
    for (system, period) in DATASETS:
        recs, qc = load_classified(system, period)
        s = summarize(recs)
        summ[(system, period)] = s
        tot = qc["agree"] + qc["disagree"]
        print(f"\n=== {system} {period}: n={qc['n']}, experimental DSSR={qc['dssr']}, "
              f"rescued from DSSR={qc['rescued']}, unannotated={qc['unannotated']}"
              + (f", DB-vs-DSSR class agreement={qc['agree']}/{tot} ({100 * qc['agree'] / tot:.0f}%)" if tot else ""))
        print(f"{'class':18s} {'n':>4s} {'len':>5s} {'TM_r':>6s} {'LDDT_r':>7s} {'%good':>6s}  families")
        for c in CLASS_ORDER:
            v = s[c]
            if not v["n"]:
                continue
            fam = ", ".join(f"{k}({n})" for k, n in v["families"].most_common(2))
            print(f"{c:18s} {v['n']:4d} {v['length']:5.0f} {v['tm']:6.3f} {v['lddt']:7.3f} {v['pct_good']:5.0f}%  {fam}")
            tsv_rows.append([system, period, c, v["n"], f"{v['length']:.0f}", f"{v['tm']:.3f}",
                             f"{v['lddt']:.3f}", f"{v['pct_good']:.1f}"])
        for key, lab in (("tm", "TM_r"), ("lddt", "LDDT_r")):
            groups = [[r[key] for r in recs if r["cls"] == c and r[key] is not None] for c in CLASS_ORDER]
            kw = kruskal(groups)
            if kw:
                print(f"  Kruskal-Wallis {lab} across classes (n>=5): H={kw[0]:.1f}, df={kw[1]}, p={kw[2]:.2g}")

    print("\n=== Generalization gap per class (median post - pre; both n >= 3) ===")
    for system in ("protein_rna", "rna_only"):
        print(f"[{system}]")
        for c in CLASS_ORDER:
            a, b = summ[(system, "post")][c], summ[(system, "pre")][c]
            if a["n"] < 3 or b["n"] < 3:
                continue
            print(f"  {c:18s} n pre/post={b['n']:3d}/{a['n']:3d}  "
                  f"LDDT_r {b['lddt']:.3f}->{a['lddt']:.3f} ({a['lddt'] - b['lddt']:+.3f})  "
                  f"%good {b['pct_good']:.0f}->{a['pct_good']:.0f}  "
                  f"TM_r {b['tm']:.3f}->{a['tm']:.3f} ({a['tm'] - b['tm']:+.3f})")

    tsv = os.path.join(outdir, "rna_structure_classes.tsv")
    with open(tsv, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["system", "period", "class", "n", "median_length", "median_TM_r",
                    "median_LDDT_r", "pct_good_RNA"])
        w.writerows(tsv_rows)
    print(f"\nSummary written to {tsv}")


if __name__ == "__main__":
    main()
