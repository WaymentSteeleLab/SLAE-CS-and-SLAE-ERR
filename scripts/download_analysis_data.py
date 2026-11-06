#!/usr/bin/env python
"""Download the data of the analysis notebooks (analysis/) and of the BMRB curation (curation/) into data/.

    python scripts/download_analysis_data.py [--parts figures,curation] [--dest DIR]
                                             [--zenodo-record ID | --zenodo-dir DIR] [--hf-dataset DIR]
                                             [--only NAME ...] [--keep-archives] [--verify]

    --parts          figures:  analysis/ inputs and intermediates, MD trajectories, re-referenced BMRB data (about 13 GB unpacked)
                     curation: BMRB dump and AlphaFold2 models used as curation/ inputs (about 6 GB unpacked; 9 GB with the two dump folders on a filesystem without hard links)
    --zenodo-dir     use a folder that already holds the Zenodo files instead of downloading them
    --hf-dataset     use an existing copy of the Hugging Face dataset; default: hf_data/ if present, otherwise download
    --only           only these archives (names in scripts/zenodo_files.csv)
    --verify         after unpacking, check the sha256 of every file against zenodo_inventory.csv.gz and the index of the HF structures

Every archive is checked against scripts/zenodo_files.csv before it is unpacked at --dest (default: the repository
root). Files that are also in the Hugging Face dataset (hws-lab/SLAE-CS-rBMRB) are not on Zenodo; they are linked from
hf_data/, or unpacked from its archives, as listed in scripts/hf_data_map.csv. Third-party code goes to external/
(analysis/DATA.md).
"""
import argparse
import csv
import gzip
import hashlib
import os
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

ZENODO_RECORD = None   # set when the record is published
DATA_REPO = "hws-lab/SLAE-CS-rBMRB"
STRUCTURE_INDEX = "rBMRB/structures_rBMRB/index.csv"
SCRIPTS = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS.parent


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(name, a, tmp):
    """Path of a Zenodo file: from --zenodo-dir, or downloaded into tmp."""
    if a.zenodo_dir:
        p = a.zenodo_dir / name
        if not p.exists():
            sys.exit(f"{p} not found")
        return p
    record = a.zenodo_record or ZENODO_RECORD
    if not record:
        sys.exit("the Zenodo record is not published yet: give --zenodo-record or --zenodo-dir")
    tmp.mkdir(parents=True, exist_ok=True)
    p = tmp / name
    if not p.exists():
        url = f"https://zenodo.org/records/{record}/files/{name}?download=1"
        print(f"downloading {url}", flush=True)
        part = p.with_name(name + ".part")
        with urllib.request.urlopen(url) as r, open(part, "wb") as fh:
            shutil.copyfileobj(r, fh, 1 << 20)
        part.rename(p)
    return p


def unpack(row, a, dest, tmp, marks):
    mark = marks / (row["file"] + ".sha256")
    if mark.exists() and mark.read_text().strip() == row["sha256"]:
        print(f"{row['file']}: already unpacked")
        return
    p = fetch(row["file"], a, tmp)
    if sha256(p) != row["sha256"]:
        sys.exit(f"{row['file']}: sha256 mismatch")
    print(f"{row['file']}: sha256 OK, unpacking {row['n_files']} files", flush=True)
    with tarfile.open(p, "r:gz") as tf:
        if hasattr(tarfile, "data_filter"):
            tf.extractall(dest, filter="data")
        else:
            tf.extractall(dest)
    mark.write_text(row["sha256"] + "\n")
    if not a.zenodo_dir and not a.keep_archives:
        p.unlink()


def link(src, dst):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    elif dst.exists():
        sys.exit(f"{dst} exists and is a directory; remove it first")
    dst.symlink_to(src)


def link_hf(a, dest):
    rows = list(csv.DictReader(open(SCRIPTS / "hf_data_map.csv")))
    hf = (a.hf_dataset or dest / "hf_data").resolve()
    if not all((hf / p).exists() for p in [r["hf_path"] for r in rows] + [STRUCTURE_INDEX]):
        if a.hf_dataset:
            sys.exit(f"{hf} does not hold every file of scripts/hf_data_map.csv")
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            sys.exit("huggingface_hub is required: pip install huggingface-hub==0.36.0")
        patterns = [r["hf_path"] for r in rows] + [STRUCTURE_INDEX]
        snapshot_download(DATA_REPO, repo_type="dataset", local_dir=hf, allow_patterns=patterns)
    with open(hf / STRUCTURE_INDEX) as fh:
        index = list(csv.DictReader(fh))
    n = 0
    for r in rows:
        if r["kind"] == "file":
            link(hf / r["hf_path"], dest / r["release_path"])
            continue
        n += 1                                # kind "archive": <split>.tar.gz, unpacked into release_path
        out, split = dest / r["release_path"], Path(r["hf_path"]).name.removesuffix(".tar.gz")
        if out.is_symlink():
            out.unlink()
        out.mkdir(parents=True, exist_ok=True)
        with tarfile.open(hf / r["hf_path"], "r:gz") as tf:
            if hasattr(tarfile, "data_filter"):
                tf.extractall(out, filter="data")
            else:
                tf.extractall(out)
        want = {x["file"]: x["sha256"] for x in index if x["split"] == split}
        if {p.name for p in out.iterdir()} != set(want):
            sys.exit(f"{out}: files differ from {STRUCTURE_INDEX}")
        if a.verify:
            bad = [name for name, h in want.items() if sha256(out / name) != h]
            if bad:
                sys.exit(f"{out}: {len(bad)} files differ from {STRUCTURE_INDEX}, e.g. {bad[0]}")
    print(f"linked {len(rows) - n} files and unpacked {n} archives from {hf}")


def split_dump(dest):
    """The curation filters move entries out of bmrb_entries/, and the two curation runs start from the same dump."""
    dump = dest / "data/curation/bmrb_dump"
    for run in ("rBMRB", "whole_BMRB"):
        out = dest / "data/curation" / run / "bmrb_entries"
        if out.exists():
            print(f"{out}: exists, left as it is (delete it to start the filters again)")
            continue
        def hard(s, d):
            try:
                os.link(s, d)
            except OSError:
                shutil.copy2(s, d)
        shutil.copytree(dump, out, copy_function=hard)
        print(f"{out}: {sum(1 for _ in out.iterdir())} entries")


def verify(a, dest, tmp, names):
    inv = fetch("zenodo_inventory.csv.gz", a, tmp)
    n = bad = 0
    with gzip.open(inv, "rt") as fh:
        for r in csv.DictReader(fh):
            if r["archive"] in names:
                n += 1
                p = dest / r["path"]
                if not p.exists() or sha256(p) != r["sha256"]:
                    bad += 1
                    print(f"differs: {r['path']}")
    print(f"verified {n} files, {bad} differ")
    if bad:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", default="figures,curation")
    ap.add_argument("--dest", type=Path, default=REPO_ROOT)
    ap.add_argument("--zenodo-record")
    ap.add_argument("--zenodo-dir", type=Path)
    ap.add_argument("--hf-dataset", type=Path)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--keep-archives", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    parts = {p for p in a.parts.split(",") if p}
    assert parts <= {"figures", "curation"}, parts
    dest = a.dest.resolve()
    tmp, marks = dest / "zenodo_download", dest / "data/.zenodo"
    marks.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(open(SCRIPTS / "zenodo_files.csv")) if r["part"] in parts and (not a.only or r["file"] in a.only)]
    for r in rows:
        unpack(r, a, dest, tmp, marks)
    if "figures" in parts and not a.only:
        link_hf(a, dest)
    if any(r["file"] == "curation_bmrb_dump.tar.gz" for r in rows):
        split_dump(dest)
    if a.verify:
        verify(a, dest, tmp, {r["file"] for r in rows})
    print("done")


if __name__ == "__main__":
    main()
