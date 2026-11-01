"""
Generic Nature-style figure composer.

Combine specific named plots ("panels") from one or more analysis/evaluation
scripts into single publication-ready figures (3-6 panels typical, 1-8 supported),
using plots.publication_figure.PublicationFigure for the shared styling, panel
letters and PNG + vector-PDF output.

Two ways to drive it
--------------------
1. Inline — one figure from one script:

    python -m plots.composite_figures <script> <panel> <panel> [...] [options]

   e.g.
    python -m plots.composite_figures phCorrelation \\
        charged_residue_occurrences_by_ph_all_chains \\
        top_mispredicted_residues_by_ph_all_chains

2. From a spec file — one or many figures, possibly mixing scripts:

    python -m plots.composite_figures --spec figures.txt
    python -m plots.composite_figures --spec figures.json

   A ``.txt`` spec has one figure per line (``#`` comments, blank lines ignored):

       # script   panel ...                                  [out:NAME ncols:N]
       phCorrelation charged_residue_occurrences_by_ph_all_chains top_mispredicted_residues_by_ph_all_chains out:ph_residues
       phCorrelation dockq_by_ph dockq_quality_class_by_ph out:ph_dockq ncols:2

   A ``.json`` spec is a single figure object or a list of them, each with keys:
       script (str, required), panels (list[str], required),
       out, ncols, single_chain (bool), msa ('+MSA'/'-MSA'), db (str)

Options (inline mode, also defaults for spec mode):
    --single-chain, --msa {+MSA,-MSA}, --db PATH, --ncols N, --out NAME,
    --outdir DIR, --list

Contract
--------
``<script>`` is resolved to a module exposing a module-level

    composite_panels(names=None, single_chain_only=False, msa_option=None,
                     db_path=None) -> {base_name: (draw_fn, title)}

where ``draw_fn(ax)`` renders one panel onto a Matplotlib Axes. A script may also
expose ``COMPOSITE_PANEL_NAMES`` (used by ``--list`` and error messages). Panel
names may be passed with or without the saved-PNG suffix (e.g. ``_all_chains``).
"""

import argparse
import importlib
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from plots.publication_figure import PublicationFigure, normalize_panel_name

# Packages searched when resolving a bare script name to a module.
_CANDIDATE_PKGS = ("", "evaluation.", "rnaCompete.", "plots.")


def _import_script(name):
    """Resolve 'phCorrelation', 'phCorrelation.py' or 'evaluation.phCorrelation'."""
    base = os.path.basename(str(name))
    if base.endswith(".py"):
        base = base[:-3]
    tried = []
    for pkg in _CANDIDATE_PKGS:
        mod_name = f"{pkg}{base}"
        tried.append(mod_name)
        try:
            return importlib.import_module(mod_name)
        except ModuleNotFoundError as exc:
            # Only swallow "module not found"; surface real import errors.
            if exc.name not in (mod_name, base, pkg.rstrip(".")):
                raise
    raise SystemExit(f"Could not import script '{name}'. Tried: {', '.join(tried)}")


def _available_names(module):
    names = getattr(module, "COMPOSITE_PANEL_NAMES", None)
    if names:
        return list(names)
    try:
        return list(module.composite_panels())
    except Exception:
        return []


# --------------------------------------------------------------------------- #
# Spec-file parsing
# --------------------------------------------------------------------------- #
def _parse_txt_spec(path):
    """Each non-comment line -> figure dict. Tokens 'out:NAME'/'ncols:N' are options."""
    figures = []
    with open(path) as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            tokens = line.split()
            script = tokens[0]
            panels, opts = [], {}
            for tok in tokens[1:]:
                if ":" in tok and tok.split(":", 1)[0] in ("out", "ncols", "msa", "db"):
                    key, val = tok.split(":", 1)
                    opts[key] = val
                elif tok in ("single_chain", "--single-chain"):
                    opts["single_chain"] = True
                else:
                    panels.append(tok)
            if not panels:
                raise SystemExit(f"{path}:{lineno}: no panels listed for '{script}'")
            fig = {"script": script, "panels": panels}
            if "out" in opts:
                fig["out"] = opts["out"]
            if "ncols" in opts:
                fig["ncols"] = int(opts["ncols"])
            if "msa" in opts:
                fig["msa"] = opts["msa"]
            if "db" in opts:
                fig["db"] = opts["db"]
            if opts.get("single_chain"):
                fig["single_chain"] = True
            figures.append(fig)
    return figures


def _parse_json_spec(path):
    with open(path) as fh:
        data = json.load(fh)
    if isinstance(data, dict) and "figures" in data:
        data = data["figures"]
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise SystemExit(f"{path}: expected a figure object or list of them")
    return data


def _load_spec(path):
    if path.endswith(".json"):
        return _parse_json_spec(path)
    return _parse_txt_spec(path)


# --------------------------------------------------------------------------- #
# Build one figure
# --------------------------------------------------------------------------- #
def _assemble(provider, panels, available_names, *, single_chain=False, msa=None,
              db=None, ncols=None, out="composite", outdir=None):
    """Core builder: resolve panels from ``provider`` and save one figure."""
    providers = provider(names=panels, single_chain_only=single_chain,
                         msa_option=msa, db_path=db)
    ordered, missing = [], []
    for raw in panels:
        key = normalize_panel_name(raw)
        if key in providers:
            ordered.append(providers[key])
        else:
            missing.append(raw)
    if missing:
        avail = ", ".join(available_names) or "(none)"
        raise SystemExit(
            f"Unknown panel(s): {', '.join(missing)}\nAvailable: {avail}")

    fig = PublicationFigure(ncols=ncols, panel_size=(6.4, 4.8), outdir=outdir)
    # Nature-style multipanel figures identify panels by the bold letter only;
    # the per-panel title text from the registry is intentionally dropped here.
    for draw, _title in ordered:
        fig.add_panel(draw, title=None)
    return fig.save(out)


def _build_figure(script, panels, *, single_chain=False, msa=None, db=None,
                  ncols=None, out=None, outdir=None):
    module = _import_script(script)
    if not hasattr(module, "composite_panels"):
        raise SystemExit(
            f"{module.__name__} does not expose composite_panels(); cannot compose.")
    script_base = os.path.splitext(os.path.basename(script))[0]
    return _assemble(
        module.composite_panels, panels, _available_names(module),
        single_chain=single_chain, msa=msa, db=db, ncols=ncols,
        out=out or f"{script_base}_composite", outdir=outdir)


def run_composite_cli(panels_provider, argv, *, default_out="composite"):
    """Drive composition from a script's *own* CLI: ``<script> composite <panel>...``.

    Tokens after the panel names may set options: ``out:NAME``, ``ncols:N``,
    ``msa:+MSA`` / ``+msa`` / ``-msa``, ``db:PATH``, ``single_chain``, or
    ``--list`` to print the script's available panels. Returns a process exit code.
    """
    names, opts = [], {}
    for tok in argv:
        if tok in ("--list", "list"):
            opts["list"] = True
        elif ":" in tok and tok.split(":", 1)[0] in ("out", "ncols", "msa", "db"):
            key, val = tok.split(":", 1)
            opts[key] = val
        elif tok in ("single_chain", "--single-chain"):
            opts["single_chain"] = True
        elif tok in ("+msa", "+MSA"):
            opts["msa"] = "+MSA"
        elif tok in ("-msa", "-MSA"):
            opts["msa"] = "-MSA"
        else:
            names.append(tok)

    home = sys.modules.get(getattr(panels_provider, "__module__", ""), None)
    available = list(getattr(home, "COMPOSITE_PANEL_NAMES", []) or [])
    if not available:
        try:
            available = list(panels_provider())
        except Exception:
            available = []

    if opts.get("list"):
        print("Available panels:")
        for n in available:
            print(f"  {n}")
        return 0
    if not names:
        print("usage: <script> composite <panel> [<panel> ...] "
              "[out:NAME ncols:N msa:+MSA db:PATH single_chain] | --list")
        return 2

    _assemble(
        panels_provider, names, available,
        single_chain=opts.get("single_chain", False),
        msa=opts.get("msa"), db=opts.get("db"),
        ncols=int(opts["ncols"]) if "ncols" in opts else None,
        out=opts.get("out", default_out), outdir=None)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("script", nargs="?", help="Script exposing composite_panels()")
    parser.add_argument("panels", nargs="*", help="Panel (plot) names to combine")
    parser.add_argument("--spec", default=None, help="txt/json spec of figures to build")
    parser.add_argument("--single-chain", action="store_true")
    parser.add_argument("--msa", choices=["+MSA", "-MSA"], default=None)
    parser.add_argument("--db", default=None)
    parser.add_argument("--ncols", type=int, default=None)
    parser.add_argument("--out", default=None, help="Output base filename")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--list", action="store_true",
                        help="List panels the given script exposes and exit")
    args = parser.parse_args(argv)

    # Spec-file mode (may build several figures, mixing scripts).
    if args.spec:
        figures = _load_spec(args.spec)
        if not figures:
            raise SystemExit(f"{args.spec}: no figures found")
        for fig in figures:
            _build_figure(
                fig["script"], fig["panels"],
                single_chain=fig.get("single_chain", args.single_chain),
                msa=fig.get("msa", args.msa),
                db=fig.get("db", args.db),
                ncols=fig.get("ncols", args.ncols),
                out=fig.get("out", args.out),
                outdir=fig.get("outdir", args.outdir),
            )
        return 0

    if not args.script:
        parser.error("a script is required (or use --spec FILE)")

    if args.list:
        module = _import_script(args.script)
        if not hasattr(module, "composite_panels"):
            raise SystemExit(f"{module.__name__} does not expose composite_panels().")
        print(f"Panels exposed by {module.__name__}:")
        for name in _available_names(module):
            print(f"  {name}")
        return 0

    if not args.panels:
        parser.error("at least one panel name is required (or use --list / --spec)")

    _build_figure(
        args.script, args.panels, single_chain=args.single_chain, msa=args.msa,
        db=args.db, ncols=args.ncols, out=args.out, outdir=args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
