"""Compose the SI structural-example figures, one per theme, panels lettered A, B, ... only.
Run from the repo root.  Output: 2025_NAR_AF3_Evaluation/figures2/si_ex_<theme>.png"""
import os
from PIL import Image, ImageDraw, ImageFont

SC = "results/plots/all_chains/publication_figures/site_chosen"
OUT = "2025_NAR_AF3_Evaluation/figures2"
W, GAP, MAXH = 3000, 60, 1100
FONT = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 64)
LF = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 46)       # condition labels
LS = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 32)       # their sub/superscripts
STRIP = 0.16                                                                      # label strip, fraction of panel width


def L(*segs):
    """label as segments: plain str, ('_', sub) or ('^', sup)"""
    return [x if isinstance(x, tuple) else ("", x) for x in segs]


MSA_ON, MSA_OFF = L("+MSA", ("_", "r")), L("\u2212MSA", ("_", "r"))
NO_IONS = L("No ions")


def squeezed(path, gap=0.06):
    """side-by-side structures with the empty columns between them cut down to `gap` of the width"""
    return {"squeeze": path, "gap": gap}


def pair(top, bottom, lt, lb):
    return {"pair": (top, bottom), "labels": (lt, lb)}

THEMES = {  # rows of panels; within a row, images share one height
    "torsion": [["rnafold/torsion_6XN5_track"], ["rnafold/torsion_8BMW_track"],
                ["rnafold/torsion_6XN5", "rnafold/torsion_8BMW"]],
    "pairing": [["rnafold/nonWC_8UBE_arcs"],
                ["rnafold/nonWC_8OMR_arcs"],
                ["rnafold/stacking_8S9V_arcs", "rnafold/stacking_8FUK_arcs"]],
    "elements": [["rnafold/pseudoknot_7QR3_arcs", "rnafold/pseudoknot_8T2B_arcs"],
                 ["rnafold/internal_loop_8DK7_arcs", "rnafold/internal_loop_8OPT_arcs"],
                 ["rnafold/missing_duplex_7W0D_sidebyside", squeezed("rnafold/missing_duplex_8SSW")],
                 ["rnafold/over_compact_8FZR", "rnafold/over_compact_8D97"]],
    "interface": [["interface/incorrect_8H1B_BC", "interface/incorrect_8E29_AB", "interface/slid_7VG2", "interface/slid_8S9X"],
                  ["interface/rotated_8YHE_CN", "interface/rotated_7TJ2_AG", "interface/shifted_8KDA_ID", "interface/shifted_7DLZ_AF"],
                  ["interface/distal_7v6c", "interface/distal_8JY0_AF", "interface/incorrect_site_7XSS", "interface/incorrect_site_8J62"],
                  ["interface/protein_protein_8VMA", "interface/protein_protein_7OS0"]],
    "msa": [[pair("rnafold/msa_8OLS_with", "rnafold/msa_8OLS_without", MSA_ON, MSA_OFF),       # RNA only
             pair("interface/msa_7X34_with", "interface/msa_7X34_without", MSA_ON, MSA_OFF)]],  # protein-RNA
    # 8OM0 (near-mirror of 8OLS) dropped; 7UO5 moved to the ablation figure (si_ex_ablation)
    "ions": [[pair("rnafold/ions_8K7W_without", "rnafold/ions_8K7W_with", NO_IONS, L("+3 K", ("^", "+"), ", 2 Na", ("^", "+"))),
              pair("rnafold/ions_7U4A_without", "rnafold/ions_7U4A_with", NO_IONS, L("+1 Mg", ("^", "2+"))),
              pair("interface/ions_7PDV_without", "interface/ions_7PDV_with", NO_IONS, L("+4 Zn", ("^", "2+"))),
              pair("interface/ions_8IDF_without", "interface/ions_8IDF_with", NO_IONS, L("+1 Zn", ("^", "2+")))]],
}


def load(p):
    im = Image.open(f"{SC}/{p}.png").convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255)); bg.alpha_composite(im)
    im = bg.convert("RGB")
    box = Image.eval(im.convert("L"), lambda v: 255 - v).point(lambda v: 255 if v > 8 else 0).getbbox()
    return im.crop(box)


def load_squeezed(spec):
    import numpy as np
    im = load(spec["squeeze"]); a = np.asarray(im.convert("L")) < 247
    used = a.any(axis=0); keep = np.ones(len(used), bool); maxgap = int(spec["gap"] * im.width)
    i = 0
    while i < len(used):
        if not used[i]:
            j = i
            while j < len(used) and not used[j]: j += 1
            if j - i > maxgap: keep[i + maxgap:j] = False
            i = j
        else:
            i += 1
    return Image.fromarray(np.asarray(im)[:, keep])


def load_pair(spec):
    """two conditions stacked at a common width, each under a blank strip for its label;
    returns the image and the strip centres as fractions of the height"""
    a, b = (load(p) for p in spec["pair"])
    w = max(a.width, b.width)
    a, b = (i.resize((w, int(i.height * w / i.width)), Image.LANCZOS) for i in (a, b))
    sh = int(STRIP * w)
    im = Image.new("RGB", (w, 2 * sh + a.height + b.height), "white")
    im.paste(a, (0, sh)); im.paste(b, (0, 2 * sh + a.height))
    im.info["labels"] = [(sh / 2 / im.height, spec["labels"][0]), ((1.5 * sh + a.height) / im.height, spec["labels"][1])]
    return im


def draw_label(d, cx, cy, segs):
    widths = [(LF if k == "" else LS).getlength(t) for k, t in segs]
    x = cx - sum(widths) / 2
    for (k, t), wd in zip(segs, widths):
        f = LF if k == "" else LS
        dy = {"": -26, "_": -2, "^": -32}[k]
        d.text((x, cy + dy), t, font=f, fill="black"); x += wd


GAPS = {"elements": 200}  # wider spacing where wide panels sit side by side


def compose(name, rows):
    gap = GAPS.get(name, GAP)
    placed, y, k = [], 0, 0
    loaded = [[load_pair(p) if isinstance(p, dict) and "pair" in p else load_squeezed(p) if isinstance(p, dict) else load(p)
               for p in row] for row in rows]
    natural = [min(MAXH, int((W - gap * (len(r) + 1)) / sum(i.width / i.height for i in r))) for r in loaded]
    full_h = [h for h, r in zip(natural, loaded) if len(r) == max(len(x) for x in loaded)]
    for ims, h in zip(loaded, natural):
        if len(ims) < max(len(x) for x in loaded):     # a short last row is not blown up beyond the full rows
            h = min(h, int(sorted(full_h)[len(full_h) // 2]))
        labs = [i.info.get("labels") for i in ims]
        ims = [i.resize((max(1, int(i.width * h / i.height)), h), Image.LANCZOS) for i in ims]
        for i, lb in zip(ims, labs): i.info["labels"] = lb
        free = W - sum(i.width for i in ims); g = free / (len(ims) + 1)
        x = g; y += 90
        for im in ims:
            placed.append((im, int(x), y, "ABCDEFGHIJKLMNOP"[k])); x += im.width + g; k += 1
        y += h
    canvas = Image.new("RGB", (W, y + 30), "white"); d = ImageDraw.Draw(canvas)
    for im, x, yy, lab in placed:
        canvas.paste(im, (x, yy)); d.text((x - 50, yy - 80), lab, font=FONT, fill="black")
        for fy, segs in im.info.get("labels") or []:
            draw_label(d, x + im.width / 2, yy + fy * im.height, segs)
    out = f"{OUT}/si_ex_{name}.png"; canvas.save(out, dpi=(400, 400)); print("saved", out, canvas.size)


if __name__ == "__main__":
    import sys
    for n, r in THEMES.items():
        if len(sys.argv) < 2 or n in sys.argv[1:]: compose(n, r)
