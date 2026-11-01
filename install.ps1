<#
.SYNOPSIS
  Set up AF3InterfaceEval on Windows.

.DESCRIPTION
  1. create the two conda envs from env/*.cross.yml (RNABench + ost),
  2. build US-align into evaluation/USalign/USalign.exe (needs a g++ from MSYS2/MinGW;
     native Windows compilation of US-align is limited -- WSL2 is the robust route),
  3. register the repo root in both envs (env\register_repo_path.py; no PYTHONPATH needed), then
  4. verify the tools + envs and report what is available / missing.

.PARAMETER SkipEnvs      Do not create/update the conda envs.
.PARAMETER SkipUsalign   Do not (re)build US-align.
.PARAMETER ForceUsalign  Rebuild US-align even if USalign.exe already runs.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File install.ps1
#>
[CmdletBinding()]
param(
  [switch]$SkipEnvs,
  [switch]$SkipUsalign,
  [switch]$ForceUsalign
)

$ErrorActionPreference = "Continue"
$Repo = $PSScriptRoot
Set-Location $Repo

function Log($m)  { Write-Host "[install] $m" }
function Warn($m) { Write-Host "[install] WARNING: $m" -ForegroundColor Yellow }
function Have($cmd) { [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }

Log "platform: Windows ($([System.Environment]::OSVersion.Version))"

function Conda-EnvExists($name) {
  if (-not (Have conda)) { return $false }
  $envs = conda env list 2>$null
  foreach ($line in $envs) {
    $tok = ($line -split '\s+')[0]
    if ($tok -eq $name) { return $true }
  }
  return $false
}

# --------------------------------------------------------------------------------------
# 1. conda environments (Windows always uses the cross-platform env files)
# --------------------------------------------------------------------------------------
function Create-Env($name, $yml) {
  if (-not (Test-Path $yml)) { Warn "env file $yml not found; skipping $name."; return }
  if (Conda-EnvExists $name) {
    Log "conda env '$name' already exists -- leaving it (delete it first to recreate)."
    return
  }
  $tool = if (Have mamba) { "mamba" } else { "conda" }
  Log "creating conda env '$name' from $yml (via $tool) ..."
  & $tool env create -f $yml
  if ($LASTEXITCODE -ne 0) {
    Warn "failed to create '$name' from $yml. If a version pin is unavailable on this"
    Warn "platform, relax it in $yml (see env/README.md) and re-run."
  }
}

if ($SkipEnvs) {
  Log "skipping conda env creation (-SkipEnvs)."
} elseif (-not (Have conda)) {
  Warn "conda not found on PATH; skipping env creation. Install Miniconda/Mambaforge,"
  Warn "then re-run, or create the envs manually (see env/README.md)."
} else {
  Warn "native Windows cannot solve the bioconda packages (openstructure, forgi, cd-hit,"
  Warn "viennarna, ...): env creation is expected to fail. Use WSL2 + install.sh (env/README.md)."
  Create-Env "RNABench" "env/silicon/environment-rnabench.yml"   # version pins only
  Create-Env "ost"      "env/silicon/environment-ost.yml"
}

# --------------------------------------------------------------------------------------
# 2. US-align -> evaluation/USalign/USalign.exe
# --------------------------------------------------------------------------------------
function Usalign-Runs($path) {
  if (-not (Test-Path $path)) { return $false }
  try { $out = & $path 2>&1 } catch { return $false }
  return ($out -match "(?i)usage")
}

function Build-Usalign {
  $dir = Join-Path $Repo "evaluation\USalign"
  $bin = Join-Path $dir "USalign.exe"
  $src = Join-Path $dir "USalign.cpp"
  if (-not (Test-Path $src)) { Warn "USalign.cpp not found in $dir; skipping."; return }
  if ((-not $ForceUsalign) -and (Usalign-Runs $bin)) {
    Log "US-align already present and runnable at $bin -- skipping build."
    return
  }
  $cxx = $null
  foreach ($c in @("g++", "clang++", "c++")) { if (Have $c) { $cxx = $c; break } }
  if (-not $cxx) {
    Warn "no C++ compiler (g++/clang++) on PATH. Install MSYS2/MinGW-w64 and add its"
    Warn "bin\ to PATH, then re-run; or build US-align under WSL2 (recommended) and copy"
    Warn "the binary to evaluation/USalign/USalign.exe."
    return
  }
  Log "building US-align with $cxx (native Windows build; use WSL2 if this fails) ..."
  Push-Location $dir
  & $cxx -O3 -ffast-math -o USalign.exe USalign.cpp
  Pop-Location
  if (Usalign-Runs $bin) {
    Log "US-align built OK -> $bin"
  } else {
    Warn "US-align build did not produce a runnable USalign.exe. US-align has limited"
    Warn "native-Windows support; the robust route is to build it under WSL2."
  }
}

if ($SkipUsalign) { Log "skipping US-align build (-SkipUsalign)." } else { Build-Usalign }

# --------------------------------------------------------------------------------------
# 3. PYTHONPATH
# --------------------------------------------------------------------------------------
# Register the repo root in each env (a .pth file in site-packages), so no PYTHONPATH is needed.
if (Have conda) {
  foreach ($name in @("RNABench", "ost")) {
    if (Conda-EnvExists $name) {
      conda run -n $name python (Join-Path $Repo "env\register_repo_path.py")
      if ($LASTEXITCODE -ne 0) { Warn "could not register the repo in '$name'" }
    }
  }
} else {
  Log "conda not found; add the repo root to PYTHONPATH instead, e.g.:"
  Write-Host "    `$env:PYTHONPATH = `"$Repo`""
}

# --------------------------------------------------------------------------------------
# 4. verification / summary
# --------------------------------------------------------------------------------------
Write-Host ""
Log "================= environment check ================="
function Report($k, $v) { Write-Host ("  {0,-22} {1}" -f $k, $v) }

if (Have conda) {
  if (Conda-EnvExists "RNABench") { Report "conda env RNABench" "OK" } else { Report "conda env RNABench" "MISSING (run without -SkipEnvs)" }
  if (Conda-EnvExists "ost")      { Report "conda env ost"      "OK" } else { Report "conda env ost"      "MISSING (run without -SkipEnvs)" }
} else {
  Report "conda" "NOT FOUND"
}

$bin = Join-Path $Repo "evaluation\USalign\USalign.exe"
if (Usalign-Runs $bin) { Report "US-align" "OK ($bin)" } else { Report "US-align" "MISSING / not runnable" }

if (Have x3dna-dssr) { Report "x3dna-dssr" "OK" } else { Report "x3dna-dssr" "not on PATH (needed only for +dssr metadata)" }

if ((Have conda) -and (Conda-EnvExists "RNABench")) {
  conda run -n RNABench python -c "import pymol" 2>$null
  if ($LASTEXITCODE -eq 0) { Report "PyMOL (RNABench)" "OK" } else { Report "PyMOL (RNABench)" "not importable (interface-area step will skip)" }
  conda run -n RNABench python -c "import freesasa" 2>$null
  if ($LASTEXITCODE -eq 0) { Report "freesasa (RNABench)" "OK" } else { Report "freesasa (RNABench)" "not importable (interface-area step will skip)" }
}

if ($env:SWISSMODEL_OST_TOKEN) {
  Report "SWISS-MODEL token" "set (SWISSMODEL_OST_TOKEN)"
} else {
  Report "SWISS-MODEL token" "unset (set SWISSMODEL_OST_TOKEN or pass --token; -noRBP to skip)"
}

Write-Host ""
Log "done. Run the drop-in eval, for example:"
Write-Host "    conda run -n RNABench python run_eval.py --pred .\preds --gt .\gt --out .\run --metrics basic"
