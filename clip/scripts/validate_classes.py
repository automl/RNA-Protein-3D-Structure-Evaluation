"""Check the manual RBP recognition-class annotation against UniProt Pfam/InterPro domains.

`rbp_class.tsv` is expert annotation written by hand: no motif atlas records *how* a
protein achieves specificity, only what it binds. To make that judgement auditable, each
gene is independently re-derived here from its annotated RNA-binding domains, fetched from
UniProt (`xref_pfam`, `xref_interpro`) for all annotated genes.

The domain -> recognition-mode mapping is from the fold's biophysics, fixed before
comparison:

  guide      PIWI / Argonaute: a loaded small RNA supplies the specificity.
  shape      Folds that read a defined RNA *shape* rather than bases -- the Roquin ROQ
             domain (constitutive decay element stem-loop) and the SRP subunits. These
             outrank any sequence-reading domain in the same protein, because that is the
             domain doing the recognition.
  sequence   Folds that hydrogen-bond to Watson-Crick edges: RRM, KH, CSD, PUF, CCCH and
             RanBP zinc fingers, YTH, NHL.
  structure  Folds that grip backbone or duplex geometry: dsRBD, DEAD/helicase, RNase,
             OB/S1.

Proteins with no sequence-reading domain at all default to structure-driven. That default
is wrong for non-canonical readers -- WDR33 reads AAUAAA through a WD40 propeller and
NUDT21/CFIm25 reads UGUA through a Nudix fold, neither of which the domain vocabulary
marks as sequence-reading -- and those cases are reported rather than silently accepted.

Run:  python3 clip/scripts/validate_classes.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import af3clip as A

GUIDE_PF = {"PF02171": "Piwi", "PF02170": "PAZ", "PF08699": "ArgoL1", "PF16486": "ArgoN"}
SHAPE_PF = {"PF18386": "ROQ_N", "PF21206": "ROQ", "PF14634": "zf-RING_5(Roquin)",
            "PF01922": "SRP19", "PF16969": "SRP68"}
SEQ_PF = {"PF00076": "RRM_1", "PF08777": "RRM_3", "PF14259": "RRM_5",
          "PF00013": "KH_1", "PF07650": "KH_2", "PF16367": "KH_9",
          "PF00313": "CSD", "PF00806": "PUF", "PF00642": "zf-CCCH",
          "PF00641": "zf-RanBP", "PF04146": "YTH", "PF01436": "NHL"}
STR_PF = {"PF00035": "dsrm", "PF00270": "DEAD", "PF00271": "Helicase_C", "PF04408": "HA2",
          "PF07717": "OB_NTP_bind", "PF00575": "S1", "PF00929": "RNase_T",
          "PF01424": "R3H", "PF04851": "ResIII"}
# InterPro fallback for proteins whose Pfam entries use newer split families
SEQ_IPR = {"IPR004087": "KH", "IPR000504": "RRM", "IPR000571": "zf-CCCH",
           "IPR035979": "RRM_sf", "IPR036612": "KH_sf"}
STR_IPR = {"IPR014720": "dsRBD", "IPR011545": "DEAD/DEAH", "IPR001650": "Helicase_C"}


def _ids(field: str) -> list[str]:
    return [x for x in (field or "").strip(";").split(";") if x]


def classify(pfam: list[str], interpro: list[str]) -> tuple[str, str]:
    """(recognition class, evidence). Priority: guide > shape > sequence > structure."""
    hit = lambda ids, tbl: [tbl[i] for i in ids if i in tbl]
    for ids, tbl, cls in ((pfam, GUIDE_PF, "guide"), (pfam, SHAPE_PF, "structure")):
        h = hit(ids, tbl)
        if h:
            return cls, ",".join(sorted(set(h)))
    h = hit(pfam, SEQ_PF) or hit(interpro, SEQ_IPR)
    if h:
        return "sequence", ",".join(sorted(set(h)))
    h = hit(pfam, STR_PF) or hit(interpro, STR_IPR)
    if h:
        return "structure", ",".join(sorted(set(h)))
    return "structure", "no RNA-binding domain (default)"


def main():
    manual = {r["gene"]: r["class"] for r in
              csv.DictReader(open(A.ROOT / "scripts/rbp_class.tsv"), delimiter="\t")}
    dom = {r["Gene Names (primary)"].upper(): r for r in
           csv.DictReader(open(A.ROOT / "results/uniprot_domains.tsv"), delimiter="\t")}

    agree, disagree, missing = [], [], []
    for g in sorted(manual):
        r = dom.get(g)
        if r is None:
            missing.append(g)
            continue
        cls, ev = classify(_ids(r["Pfam"]), _ids(r["InterPro"]))
        (agree if cls == manual[g] else disagree).append((g, manual[g], cls, ev))

    n = len(agree) + len(disagree)
    print(f"genes annotated              : {len(manual)}")
    print(f"with UniProt domain record   : {n}")
    print(f"  agree with manual call     : {len(agree)}")
    print(f"  disagree                   : {len(disagree)}")
    print(f"  concordance                : {len(agree)/n:.0%}" if n else "")
    if missing:
        print(f"no domain record             : {len(missing)}  {', '.join(missing)}")
    if disagree:
        # the measured specificity adjudicates: a sequence-reading protein scores high
        # against its own motif, a non-reader does not
        z = {}
        for r in csv.DictReader(open(A.ROOT / "results/entry_scores.tsv"), delimiter="\t"):
            try:
                v = float(r["z_all_exp"])
            except ValueError:
                continue
            if v == v:                      # drop unscored entries rather than averaging NaN
                z.setdefault(r["gene"], []).append(v)
        print("\ndisagreements (manual vs domain-derived), with measured specificity:")
        for g, m, d, ev in disagree:
            v = z.get(g, [])
            mz = f"{sum(v)/len(v):+.2f} (n={len(v)})" if v else "n/a"
            supports = "manual" if ((m == "sequence") == (v and sum(v)/len(v) > 1.2)) else "domain"
            print(f"  {g:<10} manual={m:<10} domain={d:<10} mean z_exp={mz:<14} "
                  f"data supports: {supports}   [{ev}]")
    return agree, disagree


if __name__ == "__main__":
    main()
