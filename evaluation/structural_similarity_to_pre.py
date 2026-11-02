"""
Structural similarity of every post-cutoff RNA to the pre-cutoff RNAs (US-align).

The matched pre/post comparison (evaluation/postpre_matched_similarity.py) pairs targets by RNA
SEQUENCE identity. A sequence-novel RNA can still adopt a fold that AF3 saw in training, so here
the comparison is repeated by STRUCTURE, as in Ludaic & Elofsson (NAR 2026; RNA TM-score > 0.7 =
structurally redundant):

  * the longest RNA chain of every post-cutoff target (protein-RNA, single-chain RNA, multi-chain
    RNA; the chain whose LDDT_r / TM_r the paper reports) is aligned with US-align (-mol RNA)
    against every RNA chain of every pre-cutoff target of all three sets pooled (the pre-cutoff
    structures we hold; all predate the AF3 training cutoff of 30 Sep 2021);
  * similarity = TM-score normalised by the POST chain (how much of the new RNA's fold is found
    in the old structure), maximised over the pre-cutoff chains; pre chains shorter than
    MIN_LEN_RATIO x the post chain cannot reach a TM-score of 0.5 and are skipped;
  * structural twin = the pre-cutoff target holding the best chain.

Analyses (results/plots/all_chains/structural_similarity/):
  A  distribution of the best TM-score per post target (>= 0.7 redundant, 0.5-0.7 same fold
     class, < 0.5 novel fold), and its cross-tab with the sequence identity of the sequence twin;
  B  structural-twin matched pairs: paired difference (post - structural twin) in LDDT_r / TM_r
     per TM bin (median, bootstrap 95% CI), mirroring the sequence-matched analysis;
  C  does the pre/post gap survive when structurally familiar post targets (TM >= 0.5) are
     removed? (post vs all pre, Mann-Whitney).
TM-scores of RNAs shorter than MIN_NT_TM nucleotides are not interpretable (d0 floor) and these
targets are reported separately.

Run in the RNABench env:  python evaluation/structural_similarity_to_pre.py [--align] [--workers N]
(--align recomputes the US-align table; otherwise the cached table is reused.)
"""

import argparse
import csv
import glob
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from evaluation.postpre_matched_similarity import (GOOD_RNA_LDDT, SYSTEMS, boot_median_ci, load)

USALIGN = os.path.join(_REPO_ROOT, "evaluation", "USalign", "USalign")
OUTDIR = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "structural_similarity")
WORK = os.path.join(OUTDIR, "chains")
ALIGN_TSV = os.path.join(OUTDIR, "usalign_post_vs_pre.tsv")
SEQ_PAIRS = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "postpre_matched",
                         "matched_pairs_{}.tsv")
MIN_NT = 5               # RNA chains shorter than this are not written
MIN_NT_TM = 20           # TM-score is reported only for post RNAs of at least this length
MIN_LEN_RATIO = 0.5      # skip pre chains shorter than this fraction of the post chain
TM_BINS = [(0.7, 1.01, "TM>=0.7"), (0.5, 0.7, "TM 0.5-0.7"), (0.0, 0.5, "TM<0.5")]


# --------------------------------------------------------------------------- #
# Structures -> one PDB file per RNA chain
# --------------------------------------------------------------------------- #
def _cif_index():
    idx = {}
    for d in glob.glob(os.path.join(_REPO_ROOT, "data", "**", "pdb"), recursive=True):
        for p in glob.glob(os.path.join(d, "*.cif")):
            idx.setdefault(os.path.basename(p)[:-4].lower(), p)
    return idx


def rna_chains(cif, out_dir, tag):
    """Write each RNA polymer chain of the first model as <tag>_<chain>.pdb (renamed chain A);
    return [(file name, auth chain, length)]."""
    import gemmi
    st = gemmi.read_structure(cif)
    st.setup_entities()
    st.remove_ligands_and_waters()
    out = []
    for ch in st[0]:
        pol = ch.get_polymer()
        if len(pol) < MIN_NT or pol.check_polymer_type() != gemmi.PolymerType.Rna:
            continue
        name = f"{tag}_{ch.name}.pdb"
        path = os.path.join(out_dir, name)
        if not os.path.exists(path):
            s = gemmi.Structure()
            m = gemmi.Model("1")
            c = gemmi.Chain("A")
            for r in pol:
                c.add_residue(r)
            m.add_chain(c)
            s.add_model(m)
            s.setup_entities()
            s.write_pdb(path)
        out.append((name, ch.name, len(pol)))
    return out


def build_chain_sets():
    idx = _cif_index()
    post_dir, pre_dir = os.path.join(WORK, "post"), os.path.join(WORK, "pre")
    os.makedirs(post_dir, exist_ok=True)
    os.makedirs(pre_dir, exist_ok=True)
    post, pre, missing = [], {}, []
    for system, cfg in SYSTEMS.items():
        for period, store in (("post", post), ("pre", pre)):
            for rec in load(cfg[period], cfg["table"], cfg["protein"]):
                pid = rec["id"][:4].lower()
                cif = idx.get(pid)
                if cif is None:
                    missing.append((system, period, pid))
                    continue
                chains = rna_chains(cif, post_dir if period == "post" else pre_dir, pid)
                if not chains:
                    missing.append((system, period, pid + " (no RNA chain >= MIN_NT)"))
                    continue
                if period == "post":
                    name, ch, n = max(chains, key=lambda t: t[2])        # longest RNA chain
                    post.append(dict(system=system, id=rec["id"], file=name, chain=ch, len=n))
                else:
                    for name, ch, n in chains:
                        pre[name] = dict(id=rec["id"], file=name, chain=ch, len=n)
    print(f"post targets: {len(post)}; pre RNA chains: {len(pre)} "
          f"({len({v['id'] for v in pre.values()})} pre targets); missing: {len(missing)}")
    for m in missing[:20]:
        print("   missing:", m)
    return post, list(pre.values())


# --------------------------------------------------------------------------- #
# US-align
# --------------------------------------------------------------------------- #
def _align_one(q, pre, list_dir):
    cands = [p for p in pre if p["len"] >= MIN_LEN_RATIO * q["len"]]
    if not cands:
        return []
    lst = os.path.join(list_dir, q["file"] + ".list")
    with open(lst, "w") as fh:
        fh.write("\n".join(p["file"] for p in cands) + "\n")
    cmd = [USALIGN, os.path.join(WORK, "post", q["file"]), "-dir2", os.path.join(WORK, "pre") + "/", lst,
           "-mol", "RNA", "-outfmt", "2"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    by_file = {p["file"]: p for p in cands}
    out = []
    for line in res.stdout.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split("\t")
        pre_file = os.path.basename(f[1].split(":")[0])
        p = by_file.get(pre_file)
        if p is None:
            continue
        out.append(dict(system=q["system"], post_id=q["id"], post_chain=q["chain"], post_len=q["len"],
                        pre_id=p["id"], pre_chain=p["chain"], pre_len=p["len"],
                        tm_post=float(f[2]), tm_pre=float(f[3]), rmsd=float(f[4])))
    return out


def run_alignments(post, pre, workers):
    list_dir = os.path.join(WORK, "lists")
    os.makedirs(list_dir, exist_ok=True)
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(lambda q: _align_one(q, pre, list_dir), post), 1):
            rows += r
            if i % 50 == 0:
                print(f"   aligned {i}/{len(post)} post chains", flush=True)
    keys = ["system", "post_id", "post_chain", "post_len", "pre_id", "pre_chain", "pre_len",
            "tm_post", "tm_pre", "rmsd"]
    with open(ALIGN_TSV, "w") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(keys)
        for r in rows:
            w.writerow([r[k] for k in keys])
    print(f"wrote {ALIGN_TSV}: {len(rows)} alignments")


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def _best_hits(pre_ids_of=None):
    """{(system, post id): best hit} over all pre chains, or (pre_ids_of given) only over the
    pre-cutoff targets of the same system -- the twin whose AF3 accuracy is comparable."""
    valid_pre = {r["id"] for cfg in SYSTEMS.values() for r in load(cfg["pre"], cfg["table"], cfg["protein"])}
    best = {}
    for r in csv.DictReader(open(ALIGN_TSV), delimiter="\t"):
        if r["pre_id"] not in valid_pre:          # e.g. 7PMQ, removed from the pre-cutoff set
            continue
        if pre_ids_of is not None and r["pre_id"] not in pre_ids_of[r["system"]]:
            continue
        key = (r["system"], r["post_id"])
        tm = float(r["tm_post"])
        if key not in best or tm > best[key]["tm"]:
            best[key] = dict(tm=tm, pre_id=r["pre_id"], post_len=int(r["post_len"]))
    return best


def _seq_twins(system):
    path = SEQ_PAIRS.format(system)
    if not os.path.exists(path):
        return {}
    return {r["post_id"]: float(r["rna_identity"]) for r in csv.DictReader(open(path), delimiter="\t")}


def _mwu(a, b):
    from scipy.stats import mannwhitneyu
    return mannwhitneyu(a, b).pvalue if len(a) and len(b) else np.nan


def analyse(post_lens):
    best = _best_hits()
    pre_ids_of = {system: {r["id"] for r in load(cfg["pre"], cfg["table"], cfg["protein"])}
                  for system, cfg in SYSTEMS.items()}
    best_same = _best_hits(pre_ids_of)
    lines, rows_out = [], []
    for system, cfg in SYSTEMS.items():
        post = {r["id"]: r for r in load(cfg["post"], cfg["table"], cfg["protein"])}
        pre_all = load(cfg["pre"], cfg["table"], cfg["protein"])
        pre_by_id = {r["id"]: r for r in pre_all}
        seq_id = _seq_twins(system)
        recs = []
        for pid, r in post.items():
            n = post_lens.get((system, pid))
            b, bs = best.get((system, pid)), best_same.get((system, pid))
            recs.append(dict(id=pid, len=n, tm=b["tm"] if b else np.nan,
                             tm_same=bs["tm"] if bs else np.nan, twin=bs["pre_id"] if bs else None,
                             lddt=r["lddt"], lddt_bb=r.get("lddt_bb"), tmr=r["tm"], seq_id=seq_id.get(pid, np.nan)))
        scored = [x for x in recs if x["len"] is not None and x["len"] >= MIN_NT_TM and np.isfinite(x["tm"])]
        short = [x for x in recs if x["len"] is not None and x["len"] < MIN_NT_TM]
        lines.append(f"\n=== {system}: {len(recs)} post targets, {len(scored)} with an RNA >= {MIN_NT_TM} nt "
                     f"(scored), {len(short)} shorter ===")
        tms = np.array([x["tm"] for x in scored])
        if len(tms):
            lines.append(f"  best TM-score to any pre-cutoff RNA: median {np.median(tms):.2f}; "
                         + ", ".join(f"{name} {np.mean((tms >= lo) & (tms < hi)) * 100:.0f}%"
                                     for lo, hi, name in TM_BINS))
        # A: cross-tab with sequence identity of the sequence twin
        for lo, hi, name in TM_BINS:
            sub = [x for x in scored if lo <= x["tm"] < hi and np.isfinite(x["seq_id"])]
            if sub:
                novel = sum(x["seq_id"] < 0.8 for x in sub)
                lines.append(f"  A {name:11s} n={len(sub):3d}: sequence twin < 80% identity for {novel} "
                             f"({100 * novel / len(sub):.0f}%)")
        # B: structural-twin matched pairs (twin = best pre target of the SAME system)
        for lo, hi, name in TM_BINS:
            pairs = [(x, pre_by_id.get(x["twin"])) for x in scored if lo <= x["tm_same"] < hi]
            pairs = [(a, b) for a, b in pairs if b is not None]
            for key, lab in (("lddt", "LDDT_r"), ("tmr", "TM_r")):
                bkey = "lddt" if key == "lddt" else "tm"
                ok = [(a[key], b[bkey]) for a, b in pairs if a[key] is not None and b[bkey] is not None]
                if not ok:
                    continue
                d = np.array([u - v for u, v in ok])
                med = float(np.median(d))
                lo_ci, hi_ci = boot_median_ci(d)
                # well-folded share on the backbone C3' LDDT (lddt_bb), as in the fold classes i-iv
                okb = [(a["lddt_bb"], b["lddt_bb"]) for a, b in pairs
                       if a.get("lddt_bb") is not None and b.get("lddt_bb") is not None]
                good = (f"; well folded (backbone) {100 * np.mean([v > GOOD_RNA_LDDT for _, v in okb]):.0f}% -> "
                        f"{100 * np.mean([u > GOOD_RNA_LDDT for u, _ in okb]):.0f}%") if key == "lddt" and okb else ""
                lines.append(f"  B {name:11s} {lab:6s} n={len(ok):3d}: twin {np.median([v for _, v in ok]):.3f}"
                             f" -> post {np.median([u for u, _ in ok]):.3f}, paired diff {med:+.3f} "
                             f"[{lo_ci:+.3f}, {hi_ci:+.3f}]{good}")
                rows_out.append(dict(system=system, analysis="structural_twin", subset=name, metric=lab,
                                     n=len(ok), diff_median=med, ci_lo=lo_ci, ci_hi=hi_ci))
        # C: gap without structurally familiar post targets
        pre_l = [r["lddt"] for r in pre_all if r["lddt"] is not None]
        pre_b = [r["lddt_bb"] for r in pre_all if r.get("lddt_bb") is not None]
        for name, sub in (("all scored post", scored),
                          ("post TM<0.5 (novel fold)", [x for x in scored if x["tm"] < 0.5]),
                          ("post TM<0.7 (Ludaic filter)", [x for x in scored if x["tm"] < 0.7]),
                          ("post TM>=0.7 (redundant)", [x for x in scored if x["tm"] >= 0.7])):
            pl = [x["lddt"] for x in sub if x["lddt"] is not None]
            pb = [x["lddt_bb"] for x in sub if x.get("lddt_bb") is not None]
            if pl:
                lines.append(f"  C {name:28s} n={len(pl):3d}: LDDT_r median {np.median(pl):.3f} vs all pre "
                             f"{np.median(pre_l):.3f} (n={len(pre_l)}), well folded (backbone) "
                             f"{100 * np.mean(np.array(pb) > GOOD_RNA_LDDT):.0f}% vs "
                             f"{100 * np.mean(np.array(pre_b) > GOOD_RNA_LDDT):.0f}%, MWU P={_mwu(pl, pre_l):.1e}")
        with open(os.path.join(OUTDIR, f"best_structural_twin_{system}.tsv"), "w") as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow(["post_id", "rna_len", "best_tm_any_pre", "best_tm_same_system", "structural_twin_same_system",
                        "seq_twin_identity",
                        "post_RNA_LDDT", "post_RNA_TM"])
            for x in recs:
                w.writerow([x["id"], x["len"], f"{x['tm']:.3f}", f"{x['tm_same']:.3f}", x["twin"], f"{x['seq_id']:.3f}",
                            x["lddt"], x["tmr"]])
    print("\n".join(lines))
    with open(os.path.join(OUTDIR, "structural_similarity_summary.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--align", action="store_true", help="(re)compute the US-align table")
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    post, pre = build_chain_sets()
    if args.align or not os.path.exists(ALIGN_TSV):
        run_alignments(post, pre, args.workers)
    analyse({(q["system"], q["id"]): q["len"] for q in post})


if __name__ == "__main__":
    main()
