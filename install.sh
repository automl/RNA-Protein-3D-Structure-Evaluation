#!/usr/bin/env bash
# =============================================================================
# install.sh -- set up AF3InterfaceEval on Linux / macOS.
#
#   1. create the two conda envs from env/*.yml (RNABench + ost; the *.cross.yml
#      variants on Linux / Apple Silicon, the exact-pin *.yml on macOS Intel),
#   2. build US-align natively into evaluation/USalign/USalign,
#   3. register the repo root in both envs (env/register_repo_path.py; no PYTHONPATH needed), then
#   4. verify the tools + envs and report what is available / missing.
#
# Usage:
#   bash install.sh [--skip-envs] [--skip-usalign] [--force-usalign] [--cross]
#
#   --skip-envs      do not create/update the conda envs
#   --skip-usalign   do not (re)build US-align
#   --force-usalign  rebuild US-align even if the bundled binary already runs
#   --cross          use the *.cross.yml env files even on macOS Intel
#
# Windows: use install.ps1 (PowerShell) instead, or run this under WSL2.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"
REPO="$PWD"

SKIP_ENVS=0; SKIP_USALIGN=0; FORCE_USALIGN=0; FORCE_CROSS=0
for a in "$@"; do
  case "$a" in
    --skip-envs) SKIP_ENVS=1 ;;
    --skip-usalign) SKIP_USALIGN=1 ;;
    --force-usalign) FORCE_USALIGN=1 ;;
    --cross) FORCE_CROSS=1 ;;
    -h|--help) awk 'NR>=3 && /^# ={5,}/{exit} NR>=3{sub(/^# ?/,""); print}' "$0"; exit 0 ;;
    *) echo "install: unknown option '$a' (see --help)"; exit 2 ;;
  esac
done

log()  { printf '[install] %s\n' "$*"; }
warn() { printf '[install] WARNING: %s\n' "$*"; }

case "$(uname -s)" in
  Linux)  OS=linux ;;
  Darwin) OS=macos ;;
  *)      OS=other ;;
esac
# Apple Silicon (arm64) also needs the cross env (build hashes are osx-64-specific).
ARCH="$(uname -m 2>/dev/null || echo unknown)"
[ "$OS" = macos ] && [ "$ARCH" != "x86_64" ] && FORCE_CROSS=1
log "platform: OS=$OS ARCH=$ARCH"

# --------------------------------------------------------------------------------------
# 1. conda environments
# --------------------------------------------------------------------------------------
conda_env_exists() { conda env list 2>/dev/null | awk '{print $1}' | grep -qx "$1"; }

create_env() {  # $1=env name  $2=yml file  [$3=conda subdir to build for, e.g. osx-64]
  local name="$1" yml="$2" subdir="${3:-}"
  if [ ! -f "$yml" ]; then warn "env file $yml not found; skipping $name."; return; fi
  if conda_env_exists "$name"; then
    log "conda env '$name' already exists -- leaving it (delete it first to recreate)."
    return
  fi
  local tool="conda"
  command -v mamba >/dev/null 2>&1 && tool="mamba"   # faster solver when present
  log "creating conda env '$name' from $yml (via $tool${subdir:+, subdir $subdir}) ..."
  if ! env ${subdir:+CONDA_SUBDIR="$subdir"} "$tool" env create -f "$yml"; then
    warn "failed to create '$name' from $yml. If a version pin is unavailable on this"
    warn "platform, relax it in $yml (see env/README.md) and re-run. If conda reports"
    warn "unaccepted channel Terms of Service, run 'conda tos accept' first."
    return
  fi
  if [ -n "$subdir" ]; then   # keep later installs into this env on the same subdir
    local prefix
    prefix="$(conda env list 2>/dev/null | awk -v n="$name" '$1==n {print $NF}')"
    [ -n "$prefix" ] && conda config --file "$prefix/.condarc" --set subdir "$subdir" \
      && log "pinned env '$name' to subdir $subdir"
  fi
}

if [ "$SKIP_ENVS" = 1 ]; then
  log "skipping conda env creation (--skip-envs)."
elif ! command -v conda >/dev/null 2>&1; then
  warn "conda not found on PATH; skipping env creation. Install Miniconda/Mambaforge,"
  warn "then re-run, or create the envs manually (see env/README.md)."
else
  if [ "$OS" = macos ] && [ "$FORCE_CROSS" = 0 ]; then
    # Intel Mac: exact version + build pins
    create_env RNABench env/intel/environment-rnabench.yml
    create_env ost      env/intel/environment-ost.yml
  elif [ "$OS" = macos ]; then
    # Apple Silicon: the pinned stack (Intel-only MKL, schrodinger PyMOL) has no arm64
    # builds, so build the same osx-64 packages and run them under Rosetta.
    /usr/bin/arch -x86_64 /usr/bin/true 2>/dev/null \
      || warn "Rosetta 2 not available; install it with: softwareupdate --install-rosetta"
    create_env RNABench env/silicon/environment-rnabench.yml osx-64
    create_env ost      env/silicon/environment-ost.yml      osx-64
  else
    # Linux: the same package versions, with Linux builds (env/linux, see the header of those files)
    create_env RNABench env/linux/environment-rnabench.yml
    create_env ost      env/linux/environment-ost.yml
  fi
fi

# --------------------------------------------------------------------------------------
# 2. US-align  ->  evaluation/USalign/USalign
# --------------------------------------------------------------------------------------
usalign_runs() {  # does the binary at $1 execute on this platform?
  [ -x "$1" ] || return 1
  "$1" 2>&1 | grep -qi "usage"    # US-align prints a usage banner with no args
}

build_usalign() {
  local dir="$REPO/evaluation/USalign" bin="$REPO/evaluation/USalign/USalign"
  if [ ! -f "$dir/USalign.cpp" ]; then warn "USalign.cpp not found in $dir; skipping."; return; fi
  if [ "$FORCE_USALIGN" = 0 ] && usalign_runs "$bin"; then
    log "US-align already present and runnable at $bin -- skipping build."
    return
  fi
  ( cd "$dir"
    if [ "$OS" = macos ]; then
      # Prefer the bundled universal-binary Makefile; fall back to a plain clang++ build.
      if command -v make >/dev/null 2>&1 && command -v clang++ >/dev/null 2>&1; then
        log "building US-align via bundled Makefile (universal binary) ..."
        make clean >/dev/null 2>&1 || true
        make || { warn "make failed; trying a plain clang++ build."; \
                  clang++ -O3 -ffast-math -o USalign USalign.cpp; }
      else
        log "building US-align with clang++/c++ ..."
        "${CXX:-c++}" -O3 -ffast-math -o USalign USalign.cpp
      fi
    else
      # Linux (and other POSIX): straight native compile.
      log "building US-align with ${CXX:-g++} ..."
      "${CXX:-g++}" -O3 -ffast-math -o USalign USalign.cpp -lm
    fi
    chmod +x USalign 2>/dev/null || true
  )
  if usalign_runs "$bin"; then
    log "US-align built OK -> $bin"
  else
    warn "US-align build did not produce a runnable binary. Ensure a C++ compiler is"
    warn "installed (g++/clang++), then re-run with --force-usalign."
  fi
}

if [ "$SKIP_USALIGN" = 1 ]; then
  log "skipping US-align build (--skip-usalign)."
else
  build_usalign
fi

# --------------------------------------------------------------------------------------
# 3. PYTHONPATH
# --------------------------------------------------------------------------------------
# Register the repo root in each env (a .pth file in site-packages), so no PYTHONPATH export is needed.
if command -v conda >/dev/null 2>&1; then
  for name in RNABench ost; do
    if conda_env_exists "$name"; then
      conda run -n "$name" python "$REPO/env/register_repo_path.py" || warn "could not register the repo in '$name'"
    fi
  done
else
  log "conda not found; add the repo root to PYTHONPATH instead, e.g.:"
  echo "    export PYTHONPATH=\"$REPO\${PYTHONPATH:+:\$PYTHONPATH}\""
fi

# --------------------------------------------------------------------------------------
# 4. verification / summary
# --------------------------------------------------------------------------------------
echo
log "================= environment check ================="
report() { printf '  %-22s %s\n' "$1" "$2"; }

if command -v conda >/dev/null 2>&1; then
  conda_env_exists RNABench && report "conda env RNABench" "OK" || report "conda env RNABench" "MISSING (run without --skip-envs)"
  conda_env_exists ost      && report "conda env ost"      "OK" || report "conda env ost"      "MISSING (run without --skip-envs)"
else
  report "conda" "NOT FOUND"
fi

usalign_runs "$REPO/evaluation/USalign/USalign" \
  && report "US-align" "OK ($REPO/evaluation/USalign/USalign)" \
  || report "US-align" "MISSING / not runnable"

if command -v x3dna-dssr >/dev/null 2>&1; then
  report "x3dna-dssr" "OK ($(command -v x3dna-dssr))"
else
  report "x3dna-dssr" "not on PATH (needed only for +dssr metadata)"
fi

# PyMOL / freesasa live inside the RNABench env; probe it non-fatally when possible.
if command -v conda >/dev/null 2>&1 && conda_env_exists RNABench; then
  if conda run -n RNABench python -c "import pymol" >/dev/null 2>&1; then
    report "PyMOL (RNABench)" "OK"
  else
    report "PyMOL (RNABench)" "not importable (interface-area step will skip)"
  fi
  if conda run -n RNABench python -c "import freesasa" >/dev/null 2>&1; then
    report "freesasa (RNABench)" "OK"
  else
    report "freesasa (RNABench)" "not importable (interface-area step will skip)"
  fi
fi

if [ -n "${SWISSMODEL_OST_TOKEN:-}" ]; then
  report "SWISS-MODEL token" "set (SWISSMODEL_OST_TOKEN)"
else
  report "SWISS-MODEL token" "unset (set SWISSMODEL_OST_TOKEN or pass --token; -noRBP to skip)"
fi

echo
log "done. Run the drop-in eval, for example:"
echo "    conda run -n RNABench python run_eval.py --pred ./preds --gt ./gt --out ./run --metrics basic"
