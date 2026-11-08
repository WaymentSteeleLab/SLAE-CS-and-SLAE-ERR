#!/usr/bin/env python
"""Download the SLAE-CS weights and the rBMRB data from Hugging Face and lay them out where the code
expects them (layout below). Needs huggingface_hub (pip install huggingface-hub==0.36.0);
while the repositories are private, log in first (`hf auth login`, or set HF_TOKEN).

    python scripts/download_from_hf.py [--revision REV] [--dest DIR] [--scheme PANAV|LACS|BASELINE|WHOLE]
                                       [--structures train,val,test] [--no-weights] [--no-data]
                                       [--local-model DIR] [--local-dataset DIR]

    --local-model / --local-dataset  use an existing copy of the model / dataset repository (e.g. a
                                     mirror, or a previous download) instead of downloading it

Layout written under --dest (default: the repository root; run the scripts from there):
    model/SLAE-CS-PANAV/seedNN/, model/SLAE-ERR/      weights (hws-lab/SLAE-CS)
    hf_data/rBMRB/...                                  the dataset as stored on Hugging Face
    data/structures/rBMRB/{train,val,test}/            structures unpacked from hf_data
    autoencoder.ckpt                                   pretrained SLAE encoder (sha256 checked)
    data/cs/{train,val,test}.json, data/cs/tag_labels.csv, data/cs/<flags>.json, data/cs/<error table>.json
    structures/<entry_ID>.pdb                          symlinks into hf_data
    {train,val,test}.list                              copied from splits/
"""
import argparse
import csv
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

MODEL_REPO = "hws-lab/SLAE-CS"
DATA_REPO = "hws-lab/rBMRB"
AE_URL = "https://media.githubusercontent.com/media/ProteinDesignLab/SLAE/078b94e/checkpoints/autoencoder.ckpt"
AE_SHA256 = "ac5bf9613209fc310145e58f1aa7ddc9111d833a131488fe30ea738d51f7bb99"
FLAGS = "bmrb_cs_data_withoffset_panav_flags_only.json"
ERR_TABLE = "train_val_test_oof_ensemble_10seed_N.json"
SCHEMES = ("PANAV", "LACS", "BASELINE", "WHOLE")
REPO_ROOT = Path(__file__).resolve().parent.parent


def link(src, dst):
    """ln -sfn: (re)point dst at src. Where symlinks are not allowed (Windows), files are copied."""
    dst = Path(dst)
    if not Path(src).exists():
        sys.exit(f"{src} not found")
    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    elif dst.exists():
        sys.exit(f"{dst} exists and is a directory; remove it first")
    try:
        dst.symlink_to(src)
    except OSError:
        if Path(src).is_dir():
            raise
        shutil.copy2(src, dst)


def extract(archive, out):
    with tarfile.open(archive) as tf:
        if hasattr(tarfile, "data_filter"):
            tf.extractall(out, filter="data")
        else:
            tf.extractall(out)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def snapshot(repo, repo_type, revision, local_dir, patterns=None):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub is required: pip install huggingface-hub==0.36.0")
    snapshot_download(repo, repo_type=repo_type, revision=revision, local_dir=local_dir, allow_patterns=patterns)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--revision", default="main", help="branch, tag or commit of the Hugging Face repositories (a commit hash exists in only one of the two: add --no-weights or --no-data)")
    ap.add_argument("--dest", type=Path, default=REPO_ROOT)
    ap.add_argument("--scheme", default="PANAV", choices=SCHEMES)
    ap.add_argument("--structures", default="train,val,test", help="splits whose structures to fetch")
    ap.add_argument("--no-weights", dest="weights", action="store_false", help="skip the SLAE-CS weights (autoencoder.ckpt is still downloaded)")
    ap.add_argument("--no-data", dest="data", action="store_false")
    ap.add_argument("--local-model", type=Path)
    ap.add_argument("--local-dataset", type=Path)
    a = ap.parse_args()

    dest = a.dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    splits = [s for s in a.structures.split(",") if s]
    print(f"revision={a.revision} scheme={a.scheme} dest={dest}")

    # ---------- weights ---------- #
    if a.weights:
        if a.local_model:
            link(a.local_model.resolve(), dest / "model")
        else:
            snapshot(MODEL_REPO, "model", a.revision, dest / "model")
    if not a.data:
        return

    # ---------- pretrained SLAE encoder (public SLAE repository) ---------- #
    ae = dest / "autoencoder.ckpt"
    if not ae.exists():
        part = ae.with_suffix(".ckpt.part")
        print(f"downloading {AE_URL}")
        with urllib.request.urlopen(AE_URL) as r, open(part, "wb") as fh:
            shutil.copyfileobj(r, fh, 1 << 20)
        part.rename(ae)
    if sha256(ae) != AE_SHA256:
        sys.exit("autoencoder.ckpt: sha256 mismatch")
    print(f"{ae}: OK")

    # ---------- dataset files needed for this scheme ---------- #
    R = "rBMRB"
    patterns = [f"{R}/rBMRB-PANAV/*.json", f"{R}/rBMRB-ERROR/*", f"{R}/tag_labels_*.csv",
                f"{R}/structures_rBMRB/train_cluster/*", f"{R}/structures_rBMRB/index.csv"]
    if a.scheme in ("LACS", "BASELINE"):
        patterns.append(f"{R}/rBMRB-{a.scheme}/*.json")
    elif a.scheme == "WHOLE":
        patterns += [f"{R}/whole_BMRB/*", f"{R}/structures_whole_BMRB_train/*"]
    patterns += [f"{R}/structures_rBMRB/{s}.tar.gz" for s in splits]
    if a.local_dataset:
        link(a.local_dataset.resolve(), dest / "hf_data")
    else:
        snapshot(DATA_REPO, "dataset", a.revision, dest / "hf_data", patterns)
    H = dest / "hf_data" / R

    # ---------- data/cs: split jsons, tag labels, flags, error-head table ---------- #
    cs = dest / "data" / "cs"
    cs.mkdir(parents=True, exist_ok=True)
    (dest / "data" / "structures" / "train_cluster").mkdir(parents=True, exist_ok=True)
    if a.scheme == "WHOLE":                  # whole-BMRB training set; val/test are the PANAV ones
        link(H / "whole_BMRB" / "train.json", cs / "train.json")
        for s in ("val", "test"):
            link(H / "rBMRB-PANAV" / f"{s}.json", cs / f"{s}.json")
        link(H / "tag_labels_whole_BMRB.csv", cs / "tag_labels.csv")
        shutil.copy(REPO_ROOT / "splits" / "whole_bmrb" / "train.list", dest / "train.list")
    else:
        for s in ("train", "val", "test"):
            link(H / f"rBMRB-{a.scheme}" / f"{s}.json", cs / f"{s}.json")
        link(H / "tag_labels_rBMRB.csv", cs / "tag_labels.csv")
        shutil.copy(REPO_ROOT / "splits" / "panav" / "train.list", dest / "train.list")
    for s in ("val", "test"):
        shutil.copy(REPO_ROOT / "splits" / "panav" / f"{s}.list", dest / f"{s}.list")
    subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "make_flags_json.py"),
                    "--splits", *[str(H / "rBMRB-PANAV" / f"{s}.json") for s in ("train", "val", "test")],
                    "--out", str(cs / FLAGS)], check=True)
    link(H / "rBMRB-ERROR" / ERR_TABLE, cs / ERR_TABLE)
    link(H / "structures_rBMRB" / "train_cluster" / "result_cluster.tsv",
         dest / "data" / "structures" / "train_cluster" / "result_cluster.tsv")

    # ---------- structures/: one flat directory of <entry_ID>.pdb ---------- #
    st = dest / "structures"
    st.mkdir(exist_ok=True)
    with open(H / "structures_rBMRB" / "index.csv") as fh:
        index = list(csv.DictReader(fh))
    for s in splits:
        X = dest / "data" / "structures" / "rBMRB" / s
        X.mkdir(parents=True, exist_ok=True)
        extract(H / "structures_rBMRB" / f"{s}.tar.gz", X)
        names = {r["file"] for r in index if r["split"] == s}
        if {f.name for f in X.glob("*.pdb")} != names:
            sys.exit(f"{X}: files differ from structures_rBMRB/index.csv")
        for name in sorted(names):
            link(X / name, st / name)
    if a.scheme == "WHOLE":
        # the archives store bmr<id>.pdb; the code (and train.list) use <id>.pdb
        X = dest / "data" / "structures" / "whole_BMRB_train"
        X.mkdir(parents=True, exist_ok=True)
        for t in sorted((H / "structures_whole_BMRB_train").glob("part-*.tar.gz")):
            extract(t, X)
        for f in sorted(X.rglob("*.pdb")):
            link(f, st / f.name.removeprefix("bmr"))

    # ---------- every listed id must have a structure ---------- #
    missing = 0
    for s in splits:
        for i in (l.strip() for l in open(dest / f"{s}.list")):
            if i and not (st / f"{i}.pdb").exists():
                print(f"missing structure: {i} ({s})")
                missing += 1
    if missing:
        sys.exit(f"{missing} structures missing")
    print(f"done: data in {cs}, structures in {st}, weights in {dest / 'model'}")


if __name__ == "__main__":
    main()
