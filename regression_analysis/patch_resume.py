#!/usr/bin/env python
"""One-off patcher: add resume/skip support to the regression analysis scripts.

After a run's out_root is computed we skip it if a `.run_complete` marker already
exists in that dir (unless FORCE_RERUN is set), and we write that marker at the
very end of a successful run. Env-var override keeps argparse untouched.

Idempotent: files already containing `.run_complete` are left alone.
"""
import os
import re

FILES = [
    "complex_level_regression_analysis_comparison_with_single_chain.py",
    "complex_level_regression_analysis_final.py",
    "complex_level_regression_analysis_pruned_features_rp1v1.py",
    "complex_level_regression_analysis_with_interface_full_feature_set.py",
    "complex_level_regression_analysis_with_interface_pruned_features_multichain_final.py",
    "single_chain_level_regression_analysis_final.py",
    "single_chain_level_regression_analysis_full_feature_sets.py",
    "single_chain_level_regression_analysis_no_interface.py",
    "single_chain_level_regression_analysis_selected_features_multichain_final.py",
    "single_chain_level_regression_analysis_selected_features_rnaonly_final.py",
    "single_chain_level_regression_analysis_selected_features_rp1v1.py",
    "single_chain_regression_analysis_comparison_with_complex.py",
]

SKIP = (
    '    if (out_root / ".run_complete").exists() and not os.environ.get("FORCE_RERUN"):\n'
    '        print(f"[resume] skip (already complete): {out_root}  (set FORCE_RERUN=1 to redo)")\n'
    '        return\n'
)
MARK = (
    '    try:\n'
    '        (out_root / ".run_complete").write_text("done", encoding="utf-8")\n'
    '    except Exception as _e:\n'
    '        print(f"[resume] marker write failed: {_e}")\n'
)
END_LINE = '    print("Outputs in:", out_root.resolve())\n'

for fn in FILES:
    with open(fn, encoding="utf-8") as f:
        s = f.read()
    if ".run_complete" in s:
        print("already patched, skip:", fn)
        continue
    has_os = re.search(r"(?m)^\s*import os(\b|,)", s) is not None
    lines = s.splitlines(keepends=True)
    out, did_skip, did_mark = [], False, False
    for ln in lines:
        if (not did_skip) and ln.lstrip().startswith("out_root = Path(args.outdir)"):
            out.append(ln)
            out.append(SKIP)
            did_skip = True
            continue
        if (not did_mark) and ln == END_LINE:
            out.append(MARK)
            out.append(ln)
            did_mark = True
            continue
        out.append(ln)
    if not (did_skip and did_mark):
        print(f"!! ANCHOR MISS in {fn}: skip={did_skip} mark={did_mark} -- NOT written")
        continue
    new = "".join(out)
    if not has_os:
        new = "import os\n" + new
        print("   (added 'import os')")
    with open(fn, "w", encoding="utf-8") as f:
        f.write(new)
    print("patched:", fn)

print("\nDone.")
