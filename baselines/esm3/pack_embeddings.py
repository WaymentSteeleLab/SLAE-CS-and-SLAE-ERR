#!/usr/bin/env python
"""Pack per-chunk embedding tarballs into one uncompressed .npz per key, with QC.

Reads every <prefix>_*.tar.gz in --src (one per structure chunk, each holding the extract_esm3.py
outputs <prefix>_<chunk>/<pdb_id>.npz), checks every structure against the PANAV
split JSONs, and writes <out_dir>/<prefix>_<key>.npz mapping pdb_id -> (L, D)
float16 for each --key. Arrays are streamed into the zip one by one (never all
in memory) and stored uncompressed so the trainer can load them quickly.

QC (per structure): sequence == JSON sequence of its split, shape (L, D),
all values finite. Anything failing QC is skipped and reported; a missing
report per split is printed at the end.

Run (CPU only, numpy only), from the repository root after scripts/download_from_hf.py:
    python baselines/esm3/pack_embeddings.py --src emb_chunks --prefix esm3 --key seqstruct --key seqonly \
        --panav data/cs --list_dir . --out_dir emb
"""
import argparse
import glob
import io
import json
import os
import sys
import tarfile
import zipfile

import numpy as np


def load_split_sequences(panav):
    """pdb_id (lower) -> (split, sequence), same id convention as the trainer."""
    out = {}
    for split in ("train", "val", "test"):
        with open(os.path.join(panav, f"{split}.json")) as f:
            d = json.load(f)
        for ref_id, entry in d["entry_ID"].items():
            pid = entry.lower().removeprefix("bmr")
            out[pid] = (split, d["sequence"][ref_id])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="dir with <prefix>_*.tar.gz chunk outputs")
    p.add_argument("--prefix", default="esm3")
    p.add_argument("--key", action="append", required=True, help="npz key to pack (repeatable)")
    p.add_argument("--panav", required=True, help="dir with {train,val,test}.json")
    p.add_argument("--list_dir", default=None, help="dir with {train,val,test}.list (default: --panav)")
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    seqs = load_split_sequences(args.panav)
    wanted = {}
    for split in ("train", "val", "test"):
        with open(os.path.join(args.list_dir or args.panav, f"{split}.list")) as f:
            wanted[split] = {l.strip().lower() for l in f if l.strip()}

    os.makedirs(args.out_dir, exist_ok=True)
    zips = {k: zipfile.ZipFile(os.path.join(args.out_dir, f"{args.prefix}_{k}.npz"), "w",
                               compression=zipfile.ZIP_STORED, allowZip64=True)
            for k in args.key}
    done, bad = set(), []

    tars = sorted(glob.glob(os.path.join(args.src, f"{args.prefix}_*.tar.gz")))
    if not tars:
        sys.exit(f"no {args.prefix}_*.tar.gz in {args.src}")
    for tpath in tars:
        n = 0
        with tarfile.open(tpath, "r:gz") as tf:
            for m in tf:
                if not (m.isfile() and m.name.endswith(".npz")):
                    continue
                pid = os.path.basename(m.name)[:-4].lower()
                z = np.load(io.BytesIO(tf.extractfile(m).read()))
                seq = str(z["sequence"])
                ref = seqs.get(pid)
                if ref is None:
                    bad.append((pid, "not in any split JSON")); continue
                if seq != ref[1]:
                    bad.append((pid, f"sequence mismatch vs {ref[0]}.json")); continue
                arrs = {k: z[k] for k in args.key}
                why = next((f"{k}: shape {a.shape}" for k, a in arrs.items()
                            if a.ndim != 2 or a.shape[0] != len(seq)), None) \
                    or next((f"{k}: non-finite" for k, a in arrs.items()
                             if not np.isfinite(a.astype(np.float32)).all()), None)
                if why:
                    bad.append((pid, why)); continue
                for k, a in arrs.items():
                    with zips[k].open(pid + ".npy", "w", force_zip64=True) as fh:
                        np.lib.format.write_array(fh, np.ascontiguousarray(a, dtype=np.float16))
                done.add(pid)
                n += 1
        print(f"{os.path.basename(tpath)}: {n} structures packed", flush=True)
    for zf in zips.values():
        zf.close()

    print(f"\npacked {len(done)} structures, keys={args.key}, dim={arrs[args.key[0]].shape[1] if done else '-'}")
    for split, ids in wanted.items():
        miss = sorted(ids - done)
        print(f"  {split}: {len(ids) - len(miss)}/{len(ids)} present"
              + (f"  missing e.g. {miss[:5]}" if miss else ""))
    if bad:
        print(f"\n{len(bad)} structures failed QC (skipped):")
        for pid, why in bad[:20]:
            print(f"  {pid}: {why}")
    for k in args.key:
        path = os.path.join(args.out_dir, f"{args.prefix}_{k}.npz")
        print(f"wrote {path} ({os.path.getsize(path) / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
