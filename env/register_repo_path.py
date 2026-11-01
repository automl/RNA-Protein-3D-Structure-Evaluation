"""Put the repo root on sys.path of the running Python environment, permanently.

Writes ``af3interfaceeval.pth`` (one line: the repo root) into the environment's site-packages,
so package-style imports (``from database.startConfig import StartConfig``) work from any script
and any working directory without exporting PYTHONPATH. Works the same on macOS, Linux and Windows.

    conda run -n RNABench python env/register_repo_path.py
    conda run -n ost      python env/register_repo_path.py
    python env/register_repo_path.py --remove     # undo for the active environment

A PYTHONPATH that is still exported takes precedence over the .pth entry, so remove old
``export PYTHONPATH=...`` lines that point at another checkout.
"""
import argparse
import os
import sys
import sysconfig

PTH_NAME = "af3interfaceeval.pth"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--remove", action="store_true", help="delete the .pth file instead")
    args = ap.parse_args()

    site_dir = sysconfig.get_paths()["purelib"]
    pth = os.path.join(site_dir, PTH_NAME)
    if args.remove:
        if os.path.exists(pth):
            os.remove(pth)
            print(f"removed {pth}")
        else:
            print(f"nothing to remove ({pth} not found)")
        return
    with open(pth, "w") as fh:
        fh.write(REPO_ROOT + "\n")
    print(f"{sys.executable}: repo root {REPO_ROOT} registered in {pth}")


if __name__ == "__main__":
    main()
