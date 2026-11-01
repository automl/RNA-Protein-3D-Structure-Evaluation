#!/usr/bin/env python3
"""
Build DSSR-style JSON files (<id>_<pdb|af>_dssr.json) for the +dssr scripts when only the DSSR web
server's "3DNA output" text (<id>_<pdb|af>_dssr.txt) is available, or nothing at all.

DSSR is run on the aligned RNA chain the -dssr step writes, aligned/<id>_<pdb|af>_aligned_rna.pdb
(the chain whose LDDT_r / TM_r is reported); the earlier DSSR JSON files were made from exactly these
files, and the converter reads the same files.

Preferred input is real DSSR JSON (x3dna-dssr -i=<file> --json -o=<id>_<kind>_dssr.json); existing
JSON files are never touched. Without the DSSR program, analyse each structure at
https://web.x3dna-dssr.org/analyze and download "3DNA output" as <id>_<pdb|af>_dssr.txt into the
dssr/ folder. When DSSR finds no base pair the download fails (HTTP 404) and no text file exists:
the structure is then treated as single-stranded (no pairs, all-dot dot-bracket). An empty dssr/
folder therefore means all structures are single-stranded.

What the JSON contains and where it comes from:
  pairs      DSSR's own base pairs from the text (nt1, nt2, bp, name, Saenger, LW, DSSR); none if
             there is no text file
  dbn        dot-bracket rebuilt from the canonical pairs (WC, Wobble); pseudoknot levels = largest
             nested set first, then the next largest, ...; '&' at chain ends and backbone breaks
  nts        nucleotides of the structure with chi, gamma and sugar pucker computed from the
             coordinates with DSSR's definitions (pucker: Altona-Sundaralingam phase in 36 deg bins)
  stacks     not available in the text output: the key is left out (stacking INF is undefined)

Checked against the DSSR JSON files of data/a2021/a2021x/protein_rna (197 structures whose aligned
RNA file has the same nucleotides; pairs taken from the JSON, as the text carries DSSR's pairs):
dot-bracket incl. '&' identical for 195/197 (the other 2, 8t29/8t2a, differ only in DSSR's arbitrary
[ ] vs { } labelling of two single-pair pseudoknots, labelled the other way by DSSR for the identical
8t2b); chi / sugar pucker / gamma classes identical for 100 / 99.8 / 99.98% of 4,854 nucleotides.
The text parser reads every pair of the 83 data/2026/2026_a2021nonX texts (= their num_pairs header),
and every paired nucleotide is found in the aligned RNA file.

Usage:  python evaluation/dssrTxt2json.py <processing folder>       (with dssr/ and aligned/)
"""
import argparse
import glob
import shutil
import json
import math
import os
import re

import gemmi

PUCKERS = ["C3'-endo", "C4'-exo", "O4'-endo", "C1'-exo", "C2'-endo", "C3'-exo", "C4'-endo", "O4'-exo",
           "C1'-endo", "C2'-exo"]
BRACKETS = ["()", "[]", "{}", "<>", "Aa", "Bb", "Cc", "Dd"]
O3P_LINK = 2.5          # A; O3'(i)-P(i+1) longer than this (or missing) = backbone break ('&')


# --------------------------------------------------------------------------- #
# 3DNA text
# --------------------------------------------------------------------------- #
_PAIR = re.compile(r"^\s*\d+\s+(\S+)\s.*#\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$")


def _nt_from_3dna(token):
    """'|1|T|G|12||||' -> (chain, resnum, icode)"""
    f = token.split("|")
    return f[2], int(f[4]), (f[5] or "").strip()


def parse_3dna_pairs(txt):
    """Base pairs of the first table (overlap areas), which lists every pair with its names."""
    pairs, started = [], False
    for line in open(txt):
        if line.lstrip().startswith("bp ") and "#" not in line and not started:
            started = True
            continue
        if started:
            if not line.strip():
                break
            m = _PAIR.match(line)
            if m:
                bp, n1, n2, saenger, lw, dssr, name = m.groups()
                pairs.append(dict(bp=bp, nt1=_nt_from_3dna(n1), nt2=_nt_from_3dna(n2),
                                  Saenger=saenger, LW=lw, DSSR=dssr, name=name))
    return pairs


# --------------------------------------------------------------------------- #
# structure
# --------------------------------------------------------------------------- #
def _dihedral(*atoms):
    return math.degrees(gemmi.calculate_dihedral(*[a.pos for a in atoms]))


def _conformation(r):
    a = lambda n: r.find_atom(n, "*")
    base = ("N9", "C4") if a("N9") is not None else ("N1", "C2")
    pts = [a(n) for n in ("O4'", "C1'") + base]
    chi = _dihedral(*pts) if all(pts) else None
    pts = [a(n) for n in ("O5'", "C5'", "C4'", "C3'")]
    gamma = _dihedral(*pts) if all(pts) else None
    ring = [a(n) for n in ("C1'", "C2'", "C3'", "C4'", "O4'")]
    phase = pucker = None
    if all(ring):
        C1, C2, C3, C4, O4 = ring
        v0, v1, v2 = _dihedral(C4, O4, C1, C2), _dihedral(O4, C1, C2, C3), _dihedral(C1, C2, C3, C4)
        v3, v4 = _dihedral(C2, C3, C4, O4), _dihedral(C3, C4, O4, C1)
        phase = math.degrees(math.atan2((v4 + v1) - (v3 + v0),
                                        2 * v2 * (math.sin(math.radians(36)) + math.sin(math.radians(72))))) % 360
        pucker = PUCKERS[int(phase // 36)]
    return chi, gamma, phase, pucker


def read_structure(path):
    """gemmi structure; PDB files written by PyMOL (aligned/) can carry malformed charge columns,
    which are dropped before parsing."""
    if path.lower().endswith(".pdb"):
        lines = [(l[:78].rstrip() if l.startswith(("ATOM", "HETATM")) else l.rstrip("\n")) for l in open(path)]
        return gemmi.read_pdb_string("\n".join(lines) + "\n")
    return gemmi.read_structure(path)


def nucleotides(cif):
    st = read_structure(cif)
    st.setup_entities()
    st.remove_ligands_and_waters()
    st.remove_hydrogens()
    out = []
    for ch in st[0]:
        for r in ch:
            info = gemmi.find_tabulated_residue(r.name)
            if (info is not None and info.is_nucleic_acid()) or (r.find_atom("C1'", "*") and r.find_atom("O4'", "*")):
                out.append((ch.name, r))
    return out


def _code(name):
    info = gemmi.find_tabulated_residue(name)
    if name in ("A", "C", "G", "U", "T", "DA", "DC", "DG", "DT", "DU"):
        return name[-1]
    c = info.one_letter_code if info is not None and info.one_letter_code.strip() not in ("", "?") else "n"
    return c.lower()


def _linked(r1, r2):
    o3, p = r1.find_atom("O3'", "*"), r2.find_atom("P", "*")
    return o3 is not None and p is not None and o3.pos.dist(p.pos) <= O3P_LINK


# --------------------------------------------------------------------------- #
# dot-bracket
# --------------------------------------------------------------------------- #
def _max_nested(pairs):
    """Largest non-crossing subset; ties keep the 5'-most pair."""
    partner = {}
    for i, j in pairs:
        partner[i], partner[j] = j, i
    pos = sorted(partner)
    idx = {p: k for k, p in enumerate(pos)}
    m = len(pos)
    dp = [[0] * (m + 1) for _ in range(m + 1)]
    take = lambda a, b: (1 + (dp[a + 1][idx[partner[pos[a]]] - 1] if a + 1 <= idx[partner[pos[a]]] - 1 else 0)
                         + (dp[idx[partner[pos[a]]] + 1][b] if idx[partner[pos[a]]] + 1 <= b else 0))
    for length in range(1, m + 1):
        for a in range(0, m - length + 1):
            b = a + length - 1
            best = dp[a + 1][b] if a + 1 <= b else 0
            if partner[pos[a]] > pos[a] and idx[partner[pos[a]]] <= b:
                best = max(best, take(a, b))
            dp[a][b] = best
    out, stack = [], [(0, m - 1)]
    while stack:
        a, b = stack.pop()
        if a > b:
            continue
        k = idx[partner[pos[a]]]
        if partner[pos[a]] > pos[a] and k <= b and take(a, b) == dp[a][b]:
            out.append((pos[a], pos[k]))
            stack += [(a + 1, k - 1), (k + 1, b)]
        else:
            stack.append((a + 1, b))
    return out


def dot_bracket(n, pairs):
    pp, used = [], set()
    for i, j in sorted({tuple(sorted(p)) for p in pairs if p[0] != p[1]}):
        if i not in used and j not in used:
            pp.append((i, j))
            used |= {i, j}
    levels, rest = [], pp
    while rest:
        L = _max_nested(rest)
        levels.append(L)
        rest = [p for p in rest if p not in set(L)]
    levels = levels[:1] + sorted(levels[1:], key=lambda L: (-len(L), min(L)))
    s = ["."] * n
    for lv, L in enumerate(levels):
        o, c = BRACKETS[min(lv, len(BRACKETS) - 1)]
        for i, j in L:
            s[i], s[j] = o, c
    return s


# --------------------------------------------------------------------------- #
def build(cif, txt=None):
    nuc = nucleotides(cif)
    key = lambda ch, r: (ch, r.seqid.num, r.seqid.icode.strip())
    index = {key(ch, r): k for k, (ch, r) in enumerate(nuc)}
    nt_id = lambda ch, num, ic: f"///{ch}/{num}" + (f"^{ic}" if ic else "")

    raw = parse_3dna_pairs(txt) if txt else []
    pairs, canon = [], []
    for k, p in enumerate(raw, 1):
        pairs.append(dict(index=k, nt1=nt_id(*p["nt1"]), nt2=nt_id(*p["nt2"]), bp=p["bp"], name=p["name"],
                          Saenger=p["Saenger"], LW=p["LW"], DSSR=p["DSSR"]))
        if p["name"] in ("WC", "Wobble") and p["nt1"] in index and p["nt2"] in index:
            canon.append((index[p["nt1"]], index[p["nt2"]]))
    sstr = dot_bracket(len(nuc), canon)

    nts, per_chain, all_seq, all_str = [], {}, [], []
    chain_counter = {}
    for k, (ch, r) in enumerate(nuc):
        chi, gamma, phase, pucker = _conformation(r)
        chain_counter[ch] = chain_counter.get(ch, 0) + 1
        code = _code(r.name)
        nts.append(dict(index=k + 1, index_chain=chain_counter[ch], chain_name=ch, nt_resnum=r.seqid.num,
                        nt_name=r.name, nt_code=code, nt_id=nt_id(*key(ch, r)), dbn=sstr[k], chi=chi,
                        gamma=gamma, phase_angle=phase, puckering=pucker))
        c = per_chain.setdefault(ch, dict(bseq=[], sstr=[]))
        c["bseq"].append(code)
        c["sstr"].append(sstr[k])
        if k and (nuc[k - 1][0] != ch or not _linked(nuc[k - 1][1], r)):
            all_seq.append("&")
            all_str.append("&")
        all_seq.append(code)
        all_str.append(sstr[k])
    dbn = {"all_chains": dict(num_nts=len(nuc), num_chars=len(all_seq), bseq="".join(all_seq), sstr="".join(all_str))}
    chains = {}
    for ch, c in per_chain.items():
        dbn[f"chain_{ch}"] = dict(num_nts=len(c["bseq"]), num_chars=len(c["bseq"]),
                                  bseq="".join(c["bseq"]), sstr="".join(c["sstr"]))
        chains[f"chain_{ch}"] = dict(num_nts=len(c["bseq"]), bseq="".join(c["bseq"]), sstr="".join(c["sstr"]))
    return dict(num_pairs=len(pairs), pairs=pairs, dbn=dbn, chains=chains, num_nts=len(nts), nts=nts,
                metadata=dict(source="evaluation/dssrTxt2json.py",
                              pairs_from=os.path.basename(txt) if txt else "none: no DSSR output, single-stranded",
                              conformation_from="coordinates (DSSR definitions)", stacks="not available"))


def convert_folder(folder, overwrite=False):
    dssr = os.path.join(folder, "dssr")
    os.makedirs(dssr, exist_ok=True)
    jobs = []
    for p in glob.glob(os.path.join(folder, "aligned", "*_aligned_rna.pdb")):
        pid, kind = os.path.basename(p).split("_")[:2]
        if kind in ("pdb", "af"):
            jobs.append((pid.lower(), kind, p))
    if not jobs:
        print(f"No aligned/*_aligned_rna.pdb in {folder}: run the -dssr step first (it writes them).")
    n = dict(kept_json=0, from_text=0, single_stranded=0, failed=0)
    for pid, kind, cif in sorted(jobs):
        out = os.path.join(dssr, f"{pid}_{kind}_dssr.json")
        # DSSR JSON shipped under the name of the aligned file (<id>_<kind>_aligned_rna_dssr.json, as in the
        # Zenodo bundle) is real DSSR output: use it under the name the +dssr scripts read
        shipped = os.path.join(dssr, f"{pid}_{kind}_aligned_rna_dssr.json")
        if os.path.exists(shipped) and (overwrite or not os.path.exists(out)):
            shutil.copyfile(shipped, out)
            n["kept_json"] += 1
            continue
        if os.path.exists(out) and not overwrite:
            try:
                if json.load(open(out)).get("metadata", {}).get("source") != "evaluation/dssrTxt2json.py":
                    n["kept_json"] += 1
                    continue
            except (json.JSONDecodeError, OSError):
                pass
        txt = os.path.join(dssr, f"{pid}_{kind}_dssr.txt")
        txt = txt if os.path.exists(txt) else None
        try:
            json.dump(build(cif, txt), open(out, "w"))
            n["from_text" if txt else "single_stranded"] += 1
        except Exception as e:                       # one bad file must not stop the folder
            print(f"  {pid}_{kind}: {e}")
            n["failed"] += 1
    print(f"DSSR JSON in {dssr}: {n['kept_json']} real DSSR JSON kept, {n['from_text']} built from 3DNA text, "
          f"{n['single_stranded']} single-stranded (no text), {n['failed']} failed")
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("folder", help="processing folder with dssr/ and aligned/")
    ap.add_argument("--overwrite", action="store_true", help="rebuild JSON files written by this script")
    convert_folder(os.path.abspath(ap.parse_args().folder), ap.parse_args().overwrite)


if __name__ == "__main__":
    main()
