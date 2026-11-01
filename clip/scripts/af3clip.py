"""Shared library: interface footprints from a2021.db/b2021.db, RBP motif PPMs, scoring.

Interface footprint
-------------------
`RNAMotif` in both tables is the set of RNA residues contacting protein, one list
entry per chain pair in `ChainIDpairList_proteinRNA`, with 1-based positions into
that pair's RNA chain sequence. A *footprint* is a maximal run of consecutive
positions, rendered back as sequence -- the structural analogue of a CLIP k-mer.

Motifs
------
MEME letter-probability matrices from the RBPBench catalog (catRAPID_omics v2.1
extended). `rbp_motif2name_ids.1.txt` maps motif ID -> gene + Experiment, so the
CLIP-derived subset can be separated from the in-vitro (RBNS/RNAcompete) one.
"""
from __future__ import annotations

import json
import math
import random
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MOTIFS = ROOT / "motifs"

# The two datasets, split on the AlphaFold3 training cutoff (2021-09-30). These two
# files are the ONLY structure databases this analysis reads:
#   post -- a2021.db, deposited after the cutoff, genuinely novel to AF3
#   pre  -- b2021.db, deposited before it, i.e. structures AF3 was trained on
# database/rbpDatabase.db is deliberately not used: it is a working copy that is not
# pinned to either dataset (it has been observed holding each of them at different
# times), so reading it would silently mislabel entries.
_CMP = ROOT.parent / "data" / "compareDatabases" / "post_pre" / "a2021_b2021_protein_rna"
DBS = {"post": _CMP / "a2021.db", "pre": _CMP / "b2021.db"}
DB = DBS["post"]        # default for helpers that take a single database

CLIP_EXPERIMENTS = {
    "eCLIP_ENCODE", "eCLIP", "PAR-CLIP", "HITS-CLIP", "iCLIP",
    "CLIP", "PAR-iCLIP", "dCLIP", "FLASH",
}
INVITRO_EXPERIMENTS = {"RBNS_ENCODE", "RNAcompete", "cisBP-RNA", "SELEX"}

_RES = re.compile(r"([ACGUNacgun])\((\d+)\)")


# --------------------------------------------------------------------------- #
# database side
# --------------------------------------------------------------------------- #
def jl(v):
    """Columns are either a bare scalar or a JSON list; normalise to a list."""
    if v is None:
        return []
    v = v.strip()
    if v.startswith("["):
        try:
            return json.loads(v)
        except Exception:
            return [v]
    return [v]


def parse_motif_field(s: str) -> list[tuple[str, int]]:
    """'C(1), G(2), U(4)' -> [('C',1), ('G',2), ('U',4)], position-sorted."""
    return sorted(
        ((m.group(1).upper(), int(m.group(2))) for m in _RES.finditer(s or "")),
        key=lambda t: t[1],
    )


def contiguous_runs(pairs, min_len: int = 4):
    """Maximal consecutive-position runs -> (start, sequence)."""
    out, cur = [], []
    for base, pos in pairs:
        if cur and pos == cur[-1][1] + 1:
            cur.append((base, pos))
        else:
            if len(cur) >= min_len:
                out.append((cur[0][1], "".join(b for b, _ in cur)))
            cur = [(base, pos)]
    if len(cur) >= min_len:
        out.append((cur[0][1], "".join(b for b, _ in cur)))
    return out


@dataclass
class Footprint:
    entry: str          # PDB ID (exp) or exp_db_id (pred)
    table: str          # 'exp' | 'pred'
    pair_index: int     # index into ChainIDpairList_proteinRNA
    protein_chain: str | None
    rna_chain: str | None
    start: int
    seq: str

    @property
    def length(self) -> int:
        return len(self.seq)


def load_footprints(table: str, min_len: int = 4, db: Path = DB) -> list[Footprint]:
    """All interface footprints in `table` ('exp' or 'pred')."""
    name = "exp_protein_rna" if table == "exp" else "pred_protein_rna"
    idcol = "PDBId" if table == "exp" else "exp_db_id"
    con = sqlite3.connect(db)
    rows = con.execute(
        f"select {idcol}, RNAMotif, ChainIDpairList_proteinRNA from {name}"
    ).fetchall()
    con.close()

    out: list[Footprint] = []
    for entry, motif_json, pair_json in rows:
        pairs = jl(pair_json)
        for i, m in enumerate(jl(motif_json)):
            pc = rc = None
            if i < len(pairs) and isinstance(pairs[i], list) and len(pairs[i]) == 2:
                pc, rc = pairs[i]
            for start, seq in contiguous_runs(parse_motif_field(m), min_len):
                out.append(Footprint(entry, table, i, pc, rc, start, seq))
    return out


def longest_run(motif_json: str) -> tuple[int, str]:
    """Longest contiguous contact run over all chain pairs, at ANY length.

    Used for display only: a run shorter than the 4 nt scoring threshold cannot be
    aligned to a PPM, but it is still a real contact stretch and reporting it as
    "none" would wrongly imply no interface.
    """
    best = (0, "")
    for m in jl(motif_json):
        for start, seq in contiguous_runs(parse_motif_field(m), min_len=1):
            if len(seq) > len(best[1]):
                best = (start, seq)
    return best


def dockq_nr(rbp_dockq, pair_list) -> float:
    """Mean RBP_DockQ over the *protein-RNA* interfaces only (DockQ_Nr).

    The whole-complex ``Dockq`` column is dominated by protein-protein interfaces in
    multi-chain assemblies and systematically overstates protein-RNA quality: in 8E3I
    it reads 0.86 while the two protein-RNA pairs score 0.66 and 0.73. ``RBP_DockQ``
    holds per-pair values as ``[{"A:B": 0.85}, ...]``; only pairs appearing in
    ``ChainIDpairList_proteinRNA`` are averaged.
    """
    allowed = set()
    for pr in jl(pair_list):
        if isinstance(pr, list) and len(pr) == 2:
            allowed.add(frozenset(str(x) for x in pr))
    s = "" if rbp_dockq is None else str(rbp_dockq).strip()
    if s in ("", "[]", "None", "null"):
        return float("nan")
    try:
        obj = json.loads(s)
    except (ValueError, TypeError):
        return float("nan")
    vals = []
    for d in (obj if isinstance(obj, list) else [obj]):
        if not isinstance(d, dict):
            continue
        for k, v in d.items():
            parts = str(k).split(":")
            if len(parts) == 2 and (not allowed or frozenset(parts) in allowed):
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    pass
    return float(np.mean(vals)) if vals else float("nan")


def af3_scalar(cell) -> float:
    """Mean of an AF3 per-chain confidence cell, e.g. ``[0.12, 0.12]`` -> 0.12.

    Several AF3 confidence columns store one value per chain as a JSON list; reading
    them with ``float()`` yields NaN and silently drops the metric from the analysis.
    """
    if cell is None:
        return float("nan")
    if isinstance(cell, (int, float)):
        return float(cell)
    s = str(cell).strip()
    if not s or s in ("None", "null"):
        return float("nan")
    try:
        v = json.loads(s)
    except (ValueError, TypeError):
        try:
            return float(s)
        except ValueError:
            return float("nan")
    if isinstance(v, list):
        vals = [float(x) for x in v if isinstance(x, (int, float))]
        return float(np.mean(vals)) if vals else float("nan")
    return float(v) if isinstance(v, (int, float)) else float("nan")


def entry_proteins(db: Path = DB) -> dict[str, list[str]]:
    """PDB ID -> UniProt entry names present in that structure."""
    con = sqlite3.connect(db)
    rows = con.execute("select PDBId, ProteinName from exp_protein_rna").fetchall()
    con.close()
    return {pdb: [n for n in jl(pn) if n] for pdb, pn in rows}


# --------------------------------------------------------------------------- #
# motif side
# --------------------------------------------------------------------------- #
@dataclass
class PPM:
    """A position *probability* matrix, as the source atlases distribute them.

    Rows are ACGT probabilities summing to 1 (atlas._sanitise enforces this); this
    is deliberately not a log-odds PWM. The background is applied later, at scoring
    time, by _logodds_matrix, so that it can be chosen per search space rather than
    baked into the stored matrix.
    """

    motif_id: str
    gene: str
    experiment: str
    mat: np.ndarray          # (w, 4), ACGT probabilities, rows sum to 1

    @property
    def width(self) -> int:
        return self.mat.shape[0]

    @property
    def is_clip(self) -> bool:
        return any(e in CLIP_EXPERIMENTS for e in self.experiment.split(";"))

    def consensus(self) -> str:
        return "".join("ACGU"[i] for i in self.mat.argmax(axis=1))


def read_meme(path: Path) -> dict[str, tuple[str, np.ndarray]]:
    """MEME letter-probability file -> {motif_id: (name, matrix)}."""
    out, mid, name, rows, reading = {}, None, None, [], False
    for line in path.read_text().splitlines():
        s = line.strip()
        if s.startswith("MOTIF"):
            if mid and rows:
                out[mid] = (name, np.array(rows, float))
            parts = s.split()
            mid = parts[1]
            name = parts[2] if len(parts) > 2 else parts[1]
            rows, reading = [], False
        elif s.startswith("letter-probability"):
            reading = True
        elif reading:
            vals = s.split()
            if len(vals) == 4:
                try:
                    rows.append([float(v) for v in vals])
                except ValueError:
                    reading = False
            else:
                reading = False
    if mid and rows:
        out[mid] = (name, np.array(rows, float))
    return out


def read_motif_index(path: Path) -> dict[str, tuple[str, str]]:
    """rbp_motif2name_ids -> {motif_id: (gene, experiment)}."""
    out = {}
    for i, line in enumerate(path.read_text().splitlines()):
        if i == 0 or not line.strip():
            continue
        f = line.split("\t")
        if len(f) >= 8:
            out[f[0]] = (f[1], f[7])
    return out


def load_ppms(meme: str = "catRAPID_omics_v2.1_6plus_motifs.weak_rounded.ext.meme",
              index: str = "rbp_motif2name_ids.1.txt") -> list[PPM]:
    mats = read_meme(MOTIFS / meme)
    idx = read_motif_index(MOTIFS / index)
    out = []
    for mid, (name, mat) in mats.items():
        gene, exp = idx.get(mid, (name, ""))
        out.append(PPM(mid, gene, exp, mat))
    return out


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
_BASE = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}


def _encode(seq: str) -> list[int]:
    return [_BASE.get(c, -1) for c in seq.upper()]


# Mononucleotide background used to turn a stored PPM into log-odds at scoring
# time. Uniform by default. Set BACKGROUND, or pass bg= to scan/normalised_score,
# to match the composition of the sequences being searched.
#
# Note the normalised score is invariant to a *uniform* background for a given
# alignment -- the constant cancels in (raw - min) / (max - min) -- so changing
# the uniform value only alters which overhang wins. A non-uniform background
# does change the scores themselves.
BACKGROUND = (0.25, 0.25, 0.25, 0.25)      # A, C, G, U


def _logodds_matrix(ppm: PPM, bg=None, pseudo=1e-3) -> np.ndarray:
    bg = BACKGROUND if bg is None else bg
    return np.log2((ppm.mat + pseudo) / (np.array(bg, float) + pseudo))


def scan(ppm: PPM, seq: str, min_overlap: int = 4, bg=None):
    """Best partial-overlap match of `ppm` against `seq`.

    The footprint is often shorter than the motif (a 5-nt contact run vs a 7-nt
    PPM), so offsets are allowed to hang off either end; only the overlapping
    columns are scored, and the score is rescaled by what those columns could
    attain. Returns (normalised in [0,1], raw log2-odds, overlap length).
    """
    idx = _encode(seq)
    w, L = ppm.width, len(idx)
    ov_max = min(w, L)
    if ov_max < min_overlap:
        return float("nan"), float("nan"), 0
    lo = _logodds_matrix(ppm, bg)
    col_hi, col_lo = lo.max(axis=1), lo.min(axis=1)

    best = (float("nan"), float("-inf"), 0)
    for off in range(-(w - min_overlap), L - min_overlap + 1):
        # motif column j aligns to seq position off + j
        j0, j1 = max(0, -off), min(w, L - off)
        if j1 - j0 < min_overlap:
            continue
        raw = 0.0
        ok = True
        for j in range(j0, j1):
            b = idx[off + j]
            if b < 0:
                ok = False
                break
            raw += lo[j, b]
        if not ok:
            continue
        hi, lw = col_hi[j0:j1].sum(), col_lo[j0:j1].sum()
        norm = (raw - lw) / (hi - lw) if hi > lw else float("nan")
        if raw > best[1]:
            best = (norm, raw, j1 - j0)
    return best


def ppm_logodds(ppm: PPM, seq: str, min_overlap: int = 4, bg=None) -> float:
    """Best raw log2-odds of `ppm` over `seq`, allowing partial overlap."""
    return scan(ppm, seq, min_overlap, bg)[1]


def normalised_score(ppm: PPM, seq: str, min_overlap: int = 4, bg=None) -> float:
    """Best partial-overlap match rescaled to [0,1] by the aligned columns' range."""
    return scan(ppm, seq, min_overlap, bg)[0]


def empirical_p(ppm: PPM, seq: str, n: int = 1000, seed: int = 0, bg=None) -> float:
    """P(shuffled footprint scores >= observed), mononucleotide shuffle."""
    obs = ppm_logodds(ppm, seq, bg=bg)
    if not math.isfinite(obs):
        return float("nan")
    rng = random.Random(seed)
    chars = list(seq)
    hits = 0
    for _ in range(n):
        rng.shuffle(chars)
        if ppm_logodds(ppm, "".join(chars), bg=bg) >= obs:
            hits += 1
    return (hits + 1) / (n + 1)
