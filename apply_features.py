"""
Process 2 — Register feature definitions with Feast.

Equivalent to running `feast apply` inside the feature repo (as done in
feast.ipynb). This reads feature_definitions.py and updates the registry,
creating the offline/online tables for the predictor and target views.

Note: `feast init` from the notebook only scaffolds the repo once and is not
repeated here — the repo already exists under feature_repo/feature_repo.
"""
import os
import subprocess
import sys

from config import FEAST_REPO_PATH


def _feast_cmd():
    """Prefer the `feast` CLI next to the current interpreter, else PATH."""
    scripts_dir = os.path.dirname(sys.executable)
    for exe in ("feast.exe", "feast"):
        candidate = os.path.join(scripts_dir, exe)
        if os.path.exists(candidate):
            return [candidate]
    return ["feast"]


def main():
    print(f"[apply_features] Running `feast apply` in {FEAST_REPO_PATH}")
    result = subprocess.run(
        _feast_cmd() + ["apply"],
        cwd=FEAST_REPO_PATH,
        check=True,
    )
    return result.returncode


if __name__ == "__main__":
    main()
