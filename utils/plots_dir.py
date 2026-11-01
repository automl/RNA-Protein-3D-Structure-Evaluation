"""Root folder for plots and plot tables.

mainExecution.py sets AF3EVAL_PLOTS_DIR=<--folder>/plots for the scripts it starts, so each processing
folder keeps its own plots; scripts run on their own (e.g. the paper analyses) keep writing to
<repo>/results/plots.
"""
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def plots_root():
    return os.environ.get("AF3EVAL_PLOTS_DIR") or os.path.join(_REPO_ROOT, "results", "plots")
