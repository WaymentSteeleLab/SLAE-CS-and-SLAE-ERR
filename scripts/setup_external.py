#!/usr/bin/env python
"""Clone the third-party code used by analysis/ and curation/ into external/, at the commits used for the release.

    python scripts/setup_external.py [--dest DIR]
"""
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PATCHES = {"makeshift": REPO_ROOT / "curation/rBMRB/makeshift_panav.patch"}   # applied after the clone
REPOS = {  # folder: (url, commit, subfolder or None)
    "makeshift": ("https://github.com/WaymentSteeleLab/makeshift.git", "1bee3eb5b4ac2b1638187dc33da94f4645fdae8f", None),
    "CSpred-ucbshift": ("https://github.com/THGLab/CSpred.git", "bb5bee7f0eac80a0a813388052656a82fbfa3f3c", None),
    "3Dn": ("https://github.com/spetti/structure_comparison.git", "71d05655d74e87f5798de6fd2a14b79e5e740a85", "3Dn_example"),
}


def clone(url, commit, dst):
    subprocess.run(["git", "clone", "-q", url, str(dst)], check=True)
    subprocess.run(["git", "-C", str(dst), "checkout", "-q", commit], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, default=REPO_ROOT)
    a = ap.parse_args()
    ext = a.dest.resolve() / "external"
    ext.mkdir(parents=True, exist_ok=True)
    for name, (url, commit, sub) in REPOS.items():
        dst = ext / name
        if dst.exists():
            print(f"{dst}: exists, left as it is")
            continue
        if sub:
            with tempfile.TemporaryDirectory() as tmp:
                clone(url, commit, Path(tmp) / "repo")
                shutil.copytree(Path(tmp) / "repo" / sub, dst)
        else:
            clone(url, commit, dst)
        if name in PATCHES:
            subprocess.run(["git", "-C", str(dst), "apply", str(PATCHES[name])], check=True)
        print(f"{dst}: {commit[:8]}")


if __name__ == "__main__":
    main()
