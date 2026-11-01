#!/usr/bin/env python3
"""
figsave.py -- shared multi-format figure saver used by every plotting script.

save_all_formats(fig, out) writes the SAME figure as PNG (600 dpi), PDF, SVG and
EPS, deriving the filenames from `out` by swapping the extension. Saving each
vector format from the live figure object (rather than converting a raster)
preserves vector quality for PDF/SVG/EPS. Each format is wrapped in try/except so
a single backend hiccup (e.g. EPS + transparency) never aborts a run.
"""
from __future__ import annotations
from pathlib import Path

# Default output formats for the whole project.
DEFAULT_FORMATS = ("png", "pdf", "svg", "eps")
PNG_DPI = 600


def save_all_formats(fig, out, dpi: int = PNG_DPI, bbox_inches="tight",
                     formats=DEFAULT_FORMATS, pad_inches=0.02, close=False):
    """Save `fig` to `out` in several formats. `out` may carry any extension
    (or none); it is replaced per format. Returns the list of written paths."""
    base = Path(out)
    base = base.with_suffix("")
    # Ensure the destination directory exists; on a fresh run the per-run
    # `pubfigs/` (etc.) subdir may not have been created yet, which previously
    # made every savefig fail with "No such file or directory".
    if base.parent:
        base.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        target = base.with_suffix("." + ext)
        try:
            kwargs = {"dpi": dpi}   # applies to PNG and to any raster (imshow)
            # content embedded in the vector formats (PDF/SVG/EPS) -- e.g. the
            # assembled composite figures, which montage panel images. Without
            # this the embedded panels are rasterised at the default ~100 dpi and
            # look blurry. Pure-vector figures are unaffected.
            if bbox_inches is not None:
                kwargs["bbox_inches"] = bbox_inches
                kwargs["pad_inches"] = pad_inches
            fig.savefig(target, **kwargs)
            written.append(target)
        except Exception as exc:  # noqa: BLE001 -- keep other formats going
            print(f"  [figsave warn] could not write {target.name}: {exc}")
    if close:
        import matplotlib.pyplot as plt
        plt.close(fig)
    return written
