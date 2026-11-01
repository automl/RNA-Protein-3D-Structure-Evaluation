"""Unified RBP motif atlas: RBPBench(catRAPID-ext) + CISBP-RNA + ATtRACT + RBPDB.

Every motif carries `source` (which atlas) and `experiment` (assay string), so the
CLIP-derived subset can be separated from the in-vitro one at analysis time.
"""
from __future__ import annotations
import csv, re, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import af3clip as A
from af3clip import PPM, read_meme, read_motif_index, MOTIFS

CLIP_RE = re.compile(r"\b(e|i|d|f|PAR-|PAR-i|HITS-)?CLIP\b|CLASH|FLASH|crosslink", re.I)


def _clip_flag(experiment: str) -> bool:
    return bool(CLIP_RE.search(experiment or ""))


class Motif(PPM):
    pass


def _sanitise(mat):
    """Force a valid probability matrix: finite, non-negative, rows summing to 1."""
    m = np.asarray(mat, float)
    m[~np.isfinite(m)] = 0.0
    m = np.clip(m, 0.0, None)
    rs = m.sum(axis=1, keepdims=True)
    bad = (rs[:, 0] <= 0)
    m[bad] = 0.25
    rs = m.sum(axis=1, keepdims=True)
    return m / rs


def _mk(mid, gene, exp, mat, source):
    p = PPM(mid, gene.upper(), exp, _sanitise(mat))
    p.source = source                      # type: ignore[attr-defined]
    p.clip = _clip_flag(exp)               # type: ignore[attr-defined]
    return p


def load_rbpbench():
    mats = read_meme(MOTIFS / "catRAPID_omics_v2.1_6plus_motifs.weak_rounded.ext.meme")
    idx = read_motif_index(MOTIFS / "rbp_motif2name_ids.1.txt")
    out = []
    for mid, (name, mat) in mats.items():
        gene, exp = idx.get(mid, (name, ""))
        p = _mk(mid, gene, exp, mat, "RBPBench")
        p.clip = any(e in A.CLIP_EXPERIMENTS for e in exp.split(";"))  # curated vocab
        out.append(p)
    return out


def load_cisbp():
    mats = read_meme(MOTIFS / "cisbp_rna_human.meme")
    return [_mk(mid, name, "cisBP-RNA", mat, "CISBP-RNA") for mid, (name, mat) in mats.items()]


def load_attract():
    """ATtRACT_db.txt rows -> matrices from pwm.txt (keyed by Matrix_id)."""
    ppms, cur, rows = {}, None, []
    for line in (MOTIFS / "attract/pwm.txt").read_text().splitlines():
        if line.startswith(">"):
            if cur and rows:
                ppms[cur] = np.array(rows, float)
            cur, rows = line[1:].split()[0], []
        elif line.strip():
            v = line.split()
            if len(v) == 5:
                rows.append([float(x) for x in v[1:]])
            elif len(v) == 4:
                rows.append([float(x) for x in v])
    if cur and rows:
        ppms[cur] = np.array(rows, float)

    out, seen = [], set()
    with open(MOTIFS / "attract/ATtRACT_db.txt") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["Organism"] not in ("Homo_sapiens", "Mus_musculus"):
                continue
            mid = r["Matrix_id"]
            if mid not in ppms:
                continue
            key = (r["Gene_name"].upper(), mid)
            if key in seen:
                continue
            seen.add(key)
            m = ppms[mid]
            m = m / np.clip(m.sum(axis=1, keepdims=True), 1e-9, None)
            out.append(_mk(f"ATT_{mid}", r["Gene_name"], r["Experiment_description"], m, "ATtRACT"))
    return out


def load_rbpdb():
    """RBPDB PWMDir/*.pwm -> matrices; filename stem is the RBPDB matrix id."""
    out = []
    d = MOTIFS / "rbpdb/PWMDir"
    if not d.is_dir():
        return out
    proteins = {}
    pf = MOTIFS / "rbpdb/proteins.tsv"
    if pf.exists():
        for line in pf.read_text().splitlines():
            f = line.split("\t")
            if len(f) > 4:
                proteins[f[0]] = f[4]
    for p in sorted(d.glob("*.pwm")):
        rows = []
        for line in p.read_text().splitlines():
            v = line.split()
            if len(v) == 4:
                try:
                    rows.append([float(x) for x in v])
                except ValueError:
                    pass
        if not rows:
            continue
        m = np.array(rows, float)
        if m.shape[1] != 4:
            continue
        m = m / np.clip(m.sum(axis=1, keepdims=True), 1e-9, None)
        out.append(_mk(f"RBPDB_{p.stem}", proteins.get(p.stem, p.stem), "RBPDB", m, "RBPDB"))
    return out


def load_all():
    out = []
    for fn in (load_rbpbench, load_cisbp, load_attract, load_rbpdb):
        try:
            got = fn()
            out += got
            print(f"  {fn.__name__:<16} {len(got):>5} motifs", file=sys.stderr)
        except Exception as e:
            print(f"  {fn.__name__:<16} FAILED: {e}", file=sys.stderr)
    return out


if __name__ == "__main__":
    ms = load_all()
    import collections
    print(f"\ntotal {len(ms)} motifs, {len(set(m.gene for m in ms))} genes, "
          f"{sum(m.clip for m in ms)} CLIP-derived "
          f"({len(set(m.gene for m in ms if m.clip))} genes)")
    print(collections.Counter(m.source for m in ms))
