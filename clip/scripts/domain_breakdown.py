"""Motif agreement and interface quality of the atlas entries, grouped by the RBP's own InterPro domains.

Domains come from the benchmark's InterPro annotation (exp_protein_rna.domain_ids / domain_names, pooled over the
protein chains of a complex; database/fetchInterProDomains.py). A complex domain is credited to the entry's gene when
its InterPro accession is among the gene's UniProt InterPro entries (results/uniprot_domains.tsv), or when all protein
chains of the complex share one sequence (homomers, where the accessions of the two sources can differ). Partner
proteins' domains are therefore not credited to the RBP. Domains are grouped into families (PUF, cold-shock, RRM, ...);
an entry with several families counts once in each. Summaries are weighted by entry and split by cohort, because the
atlas entries are mostly pre-cutoff.

  z_exp / z_pred   motif z of the experimental / AF3 interface, combined arm (entry_scores.tsv)
  z_clip, z_invitro   experimental z against CLIP-only / in vitro-only motifs
  contact          share of entries whose AF3 interface contacts >= half of the motif-bearing positions (positional.tsv)
  on_site          share of entries whose best-scoring AF3 motif lies on the native positions (positional.tsv)
  DockQ_Nr         mean (and number below 0.23) over the entries with a protein-RNA DockQ

Run from clip/scripts:  python domain_breakdown.py   -> ../results/domain_breakdown.{tsv,txt}
"""
import ast, json, re, sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT.parent / "data"
# Default: the databases fetch_data.sh populates. The shipped bundle under
# all_data_zip/all_databases carries a richer InterPro annotation, which changes the
# KH and zinc-finger rows, so --db-post/--db-pre (or --bundle) make the source explicit.
DBS = {"post": DATA / "compareDatabases/post_pre/a2021_b2021_protein_rna/a2021.db",
       "pre": DATA / "compareDatabases/post_pre/a2021_b2021_protein_rna/b2021.db"}
BUNDLE = {"post": DATA / "all_data_zip/all_databases/a2021_post_cutoff/protein_rna.db",
          "pre": DATA / "all_data_zip/all_databases/b2021_pre_cutoff/protein_rna.db"}
FAMILIES = [("PUF", r"PUM|Pumilio"), ("zinc finger", r"Znf|zf|Zinc"), ("RRM", r"^RRM"), ("KH", r"^KH"),
            ("dsRBD", r"dsRBD|DSRM"), ("helicase", r"Helicase|DEAD|RLR|RIG"), ("cold-shock", r"^CSD|Cold"),
            ("Argonaute", r"PAZ|Piwi|Argonaute"), ("WD40", r"WD40")]


def family(name):
    return next((f for f, p in FAMILIES if re.search(p, name, re.I)), "other: " + name)


def _list(v):
    for parse in (json.loads, ast.literal_eval):
        try:
            x = parse(v)
            return x if isinstance(x, list) else [x]
        except Exception:
            pass
    return [v]


def entries():
    E = pd.read_csv(ROOT / "results/entry_scores.tsv", sep="\t").drop(columns=["DockQ_Nr"], errors="ignore")
    P = pd.read_csv(ROOT / "results/positional.tsv", sep="\t")
    E = E.merge(P[["gene", "pdb", "cohort", "same_site", "recall_pair", "DockQ_Nr"]], on=["gene", "pdb", "cohort"], how="left")
    U = pd.read_csv(ROOT / "results/uniprot_domains.tsv", sep="\t")
    ipr = {r["Gene Names (primary)"]: set(str(r["InterPro"]).strip(";").split(";")) for _, r in U.iterrows()}
    dom = {}
    for cohort, db in DBS.items():
        for pid, ids, names, seqs in sqlite3.connect(db).execute(
                "SELECT PDBId, domain_ids, domain_names, ProteinSequence FROM exp_protein_rna"):
            if ids and ids != "-":
                dom[(cohort, pid.strip().upper())] = (list(zip(ids.split(";"), names.split(";"))),
                                                      len(set(map(str, _list(seqs)))) == 1)
    rows = []
    for _, r in E.iterrows():
        ds, homomer = dom.get((r.cohort, r.pdb.upper()), ([], False))
        for f in {family(n) for i, n in ds if i in ipr.get(r.gene, set()) or homomer}:
            rows.append(dict(r, family=f))
    return E, pd.DataFrame(rows)


def summarise(d):
    dq = d.DockQ_Nr.dropna()
    return pd.Series(dict(entries=len(d), genes=d.gene.nunique(), z_exp=d.z_all_exp.mean(), z_pred=d.z_all_pred.mean(),
                          z_pred_ge2=int((d.z_all_pred >= 2).sum()), z_clip=d.z_clip_exp.mean(), z_invitro=d.z_invitro_exp.mean(),
                          contact=np.nanmean(d.recall_pair.dropna() >= 0.5) if d.recall_pair.notna().any() else np.nan,
                          on_site=d.same_site.mean(), dockq_mean=dq.mean(), dockq_n=len(dq), dockq_failed=int((dq < 0.23).sum())))


def main(db_post=None, db_pre=None):
    # db_post / db_pre override the protein-RNA databases the domain annotation is read from
    if db_post:
        DBS["post"] = Path(db_post)
    if db_pre:
        DBS["pre"] = Path(db_pre)
    for c, p in DBS.items():
        if not Path(p).exists():
            raise SystemExit(f"{c} database not found: {p}")
        print(f"  {c:4s} db = {p}")
    E, D = entries()
    out = []
    for f, d in D.groupby("family"):
        for cohort, s in [("all", d)] + list(d.groupby("cohort")):
            out.append(dict(family=f, cohort=cohort, genes_list=",".join(sorted(set(s.gene))), **summarise(s)))
    T = pd.DataFrame(out).sort_values(["entries"], ascending=False)
    T.to_csv(ROOT / "results/domain_breakdown.tsv", sep="\t", index=False, float_format="%.3f")
    lines = [f"{len(E)} entries ({E.gene.nunique()} genes); {D.drop_duplicates(['gene', 'pdb', 'cohort']).shape[0]} with an own annotated domain"]
    main_fams = ["PUF", "cold-shock", "RRM", "KH", "dsRBD", "helicase", "Argonaute", "WD40", "zinc finger"]
    pd.set_option("display.width", 250)
    lines.append(T[T.family.isin(main_fams)].drop(columns="genes_list").round(2).to_string(index=False))
    seq = D[D.family.isin(["PUF", "cold-shock", "RRM"])].drop_duplicates(["gene", "pdb", "cohort"])
    post = D[D.family.isin(["cold-shock", "RRM"]) & (D.cohort == "post")].drop_duplicates(["gene", "pdb", "cohort"])
    lines.append(f"\nPUF, cold-shock or RRM: {len(seq)} entries from {seq.gene.nunique()} genes")
    lines.append(f"post-cutoff RRM or cold-shock: {len(post)} entries, mean z_pred {post.z_all_pred.mean():+.2f}, "
                 f"{int(post.DockQ_Nr.notna().sum())} with a DockQ_Nr, {int((post.DockQ_Nr < 0.23).sum())} below 0.23")
    text = "\n".join(lines)
    (ROOT / "results/domain_breakdown.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db-post", default=None, help="post-cutoff protein_rna database")
    ap.add_argument("--db-pre", default=None, help="pre-cutoff protein_rna database")
    ap.add_argument("--bundle", action="store_true",
                    help="read both from data/all_data_zip/all_databases (the shipped bundle)")
    a = ap.parse_args()
    if a.bundle and (a.db_post or a.db_pre):
        raise SystemExit("--bundle cannot be combined with --db-post/--db-pre")
    if a.bundle:
        main(db_post=BUNDLE["post"], db_pre=BUNDLE["pre"])
    else:
        main(db_post=a.db_post, db_pre=a.db_pre)
