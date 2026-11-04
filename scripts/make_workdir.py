#!/usr/bin/env python
"""Make a working directory for one run. The scripts read SLAE/configs, data/cs/, structures/ and the
{train,val,test}.list files relative to the directory they are started from, and write their
checkpoints / predictions there, so every run gets its own directory with links to the shared files.

    python scripts/make_workdir.py <dir> [--data-root DIR] [--structures LIST]

    --data-root   where download_from_hf.py put the data (default: the repository root)
    --structures  link only the structures of the ids in LIST (e.g. splits/panav/test.list for
                  test-set inference, splits/kfold/heldout_fold0.list for out-of-fold inference);
                  default: all of <data-root>/structures
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def link(src, dst):
    """ln -sfn: (re)point dst at src."""
    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    elif dst.exists():
        sys.exit(f"{dst} exists and is a directory; remove it first")
    dst.symlink_to(src)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path)
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT)
    ap.add_argument("--structures", type=Path, help="id list; link only these structures")
    a = ap.parse_args()
    data_root = a.data_root.absolute()
    wd = a.dir.absolute()
    wd.mkdir(parents=True, exist_ok=True)

    link(REPO_ROOT / "SLAE", wd / "SLAE")
    link(data_root / "data", wd / "data")
    if (data_root / "autoencoder.ckpt").is_file():
        link(data_root / "autoencoder.ckpt", wd / "autoencoder.ckpt")
    for s in ("train", "val", "test"):
        if (data_root / f"{s}.list").is_file():
            link(data_root / f"{s}.list", wd / f"{s}.list")

    st = wd / "structures"
    if a.structures is None:
        link(data_root / "structures", st)
    else:
        if st.is_symlink():                      # a previous run linked the whole folder
            st.unlink()
        st.mkdir(exist_ok=True)
        n = 0
        for i in (l.strip() for l in open(a.structures)):
            if not i:
                continue
            src = data_root / "structures" / f"{i}.pdb"
            if not src.exists():
                sys.exit(f"missing structure {i}")
            link(src.resolve(), st / f"{i}.pdb")
            n += 1
        print(f"linked {n} structures")
    print(f"working directory: {wd}")


if __name__ == "__main__":
    main()
