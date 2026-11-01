#!/bin/bash
# Windows MAX_PATH workaround.
# This project sits under a deep base path, so the analysis's nested output dirs
# can exceed the Windows 260-char path limit (LongPathsEnabled is off here). We
# map a short drive letter X: to the project and run everything from there, which
# keeps every absolute path well under the limit. Reversible: `subst X: /D`.
#
# Sourced from run_all.sh (and safe to source before any analysis script) so the
# fix is applied automatically on every run, including after a reboot when the
# per-session subst mapping is gone.
# Derive THIS folder's Windows path dynamically so the mapping works wherever the
# repo is cloned (no hardcoded path). pwd -W gives the Windows path in Git Bash;
# convert forward slashes to backslashes for subst.
PROJ_WIN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -W | tr '/' '\\')"
# Force-remap X: to THIS project every run (a stale mapping to another folder would
# silently run the pipeline in the wrong place). Unmap via cmd so Git Bash does not
# mangle the /D switch into a path.
cmd //c "subst X: /D" >/dev/null 2>&1 || true
subst X: "$PROJ_WIN" 2>/dev/null || echo "[ensure_x] WARNING: could not map X: to $PROJ_WIN"
if cd /x 2>/dev/null; then
  echo "[ensure_x] running from X:\\  ->  $PROJ_WIN"
else
  echo "[ensure_x] WARNING: X: not available; staying in $(pwd) (long paths may fail)"
fi
