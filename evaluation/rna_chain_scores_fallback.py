"""Per-chain RNA LDDT_r and TM_r on the exact residue mapping, for RNA chains without a per-chain value.

Some post-cutoff RNA chains have no entry in RNA_LDDT_AllChains / RNA_TM_AllChains, mostly in complexes where
OpenStructure's chain mapping failed (poly-U RNAs). For these, the native and the AF3 chain are cut to the residues of
the exact residue mapping (as in interface_own_dockq.py) and renumbered 1..n; LDDT_r is OpenStructure's all-atom lDDT
of that chain pair (Scorer with the chain mapping fixed) and TM_r the US-align TM-score normalised by the native chain.

  export  (RNABench env)  complex maps of the requested complexes -> <OUTDIR>/rna_chain_fallback_maps.json
  score   (ost env)       per-chain values -> <OUTDIR>/rna_chain_fallback.tsv
The requested chains are (pdb, model chain) pairs in --chains <json>.
"""
import csv
import json
import os
import subprocess
import sys
import tempfile

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
OUTDIR = os.path.join(_REPO, "results", "plots", "all_chains", "interface_rna_pose")
MAPS = os.path.join(OUTDIR, "rna_chain_fallback_maps.json")
TSV = os.path.join(OUTDIR, "rna_chain_fallback.tsv")
USALIGN = os.path.join(_REPO, "evaluation", "USalign", "USalign")


def export(chains):
    from evaluation import interface_own_dockq as M
    pdbs = {p for p, _ in chains}
    M.ost_unmapped = lambda: {("post", p): set() for p in pdbs}
    empty = os.path.join(tempfile.mkdtemp(), "none.tsv")
    open(empty, "w").write("period\tpdb\n")
    M.OWN_TSV = empty
    M._complex_maps_path = lambda validate=False: MAPS
    M.export_complex(False)


def score(chains):
    from ost import io
    from evaluation.interface_own_dockq import _cut, _load, _scorer
    want = {(p, c) for p, c in chains}
    rows, cache = [], {}
    for it in json.load(open(MAPS)):
        for c in it["chains"]:
            if (it["pdb"], c["model_chain"]) not in want:
                continue
            rec = dict(period=it["period"], pdb=it["pdb"], model_chain=c["model_chain"], native_chain=c["native_chain"],
                       n_mapped=len(c["pairs"]), lddt="", tm="", error="")
            try:
                ref = _cut(_load(it["cif"], cache), [(c["native_chain"], [a for a, _ in c["pairs"]])])
                mod = _cut(_load(it["model"], cache), [(c["model_chain"], [b for _, b in c["pairs"]])])
                rec["lddt"] = round(float(_scorer(mod, ref, {c["native_chain"]: c["model_chain"]}).lddt), 4)
                with tempfile.TemporaryDirectory() as tmp:
                    a, b = os.path.join(tmp, "model.pdb"), os.path.join(tmp, "native.pdb")
                    for ent in (mod, ref):                # PDB format: single-letter chain names only (e.g. 8PDR 'BA')
                        ent.EditXCS().RenameChain(ent.chains[0], "A")
                    io.SavePDB(mod, a); io.SavePDB(ref, b)
                    out = subprocess.run([USALIGN, a, b, "-mol", "RNA"], capture_output=True, text=True).stdout
                    tm = [l for l in out.splitlines() if l.startswith("TM-score=") and "Structure_2" in l]
                    rec["tm"] = round(float(tm[0].split()[1]), 4) if tm else ""
            except Exception as e:                       # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {e}"[:150]
            rows.append(rec)
    with open(TSV, "w") as fh:
        w = csv.DictWriter(fh, list(rows[0]) if rows else ["pdb"], delimiter="\t"); w.writeheader(); w.writerows(rows)
    print(f"{sum(1 for r in rows if r['lddt'] != '')}/{len(want)} chains scored -> {TSV}")
    for r in rows:
        if r["error"]:
            print("  ", r["pdb"], r["model_chain"], r["error"])


if __name__ == "__main__":
    mode, chains = sys.argv[1], [tuple(x) for x in json.load(open(sys.argv[sys.argv.index("--chains") + 1]))]
    export(chains) if mode == "export" else score(chains)
