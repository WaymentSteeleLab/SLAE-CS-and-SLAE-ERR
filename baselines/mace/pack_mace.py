#!/usr/bin/env python
"""Pack per-chunk MACE tarballs into one uncompressed .npz for the trainer, with QC.

MACE counterpart of ../esm3/pack_embeddings.py. Reads every
mace_*.tar.gz in --src (one per structure chunk, each holding the extract_mace.py outputs
mace_<chunk>/<pdb_id>.npz with keys resnum, resname, N), keeps the structures
of the PANAV split (train/val/test .list), checks each against its
split JSON, and writes <out_dir>/mace_N.npz mapping pdb_id -> (L, 448) float32
(MACE ran in float64; float32 keeps that precision for the head).

QC (per structure): sequence rebuilt from resname == JSON sequence of its
split, shape (L, D), all values finite. Structures not in the split
lists are skipped and counted; QC failures are skipped and reported; a missing report per split is
printed at the end.

Run (CPU only, numpy only), from the repository root after scripts/download_from_hf.py:
    python baselines/mace/pack_mace.py --src emb_chunks \
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

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


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
    p.add_argument("--src", required=True, help="dir with mace_*.tar.gz chunk outputs")
    p.add_argument("--key", default="N", help="atom key to pack")
    p.add_argument("--panav", required=True, help="dir with {train,val,test}.json")
    p.add_argument("--list_dir", default=None, help="dir with {train,val,test}.list (default: --panav)")
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    seqs = load_split_sequences(args.panav)
    wanted = {}
    for split in ("train", "val", "test"):
        with open(os.path.join(args.list_dir or args.panav, f"{split}.list")) as f:
            wanted[split] = {l.strip().lower() for l in f if l.strip()}
    in_split = set().union(*wanted.values())

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"mace_{args.key}.npz")
    zf = zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True)
    done, bad, dropped, dim = set(), [], 0, None

    tars = sorted(glob.glob(os.path.join(args.src, "mace_*.tar.gz")))
    if not tars:
        sys.exit(f"no mace_*.tar.gz in {args.src}")
    for tpath in tars:
        n = 0
        with tarfile.open(tpath, "r:gz") as tf:
            for m in tf:
                if not (m.isfile() and m.name.endswith(".npz")):
                    continue
                pid = os.path.basename(m.name)[:-4].lower()
                if pid not in in_split:
                    dropped += 1; continue
                z = np.load(io.BytesIO(tf.extractfile(m).read()))
                seq = "".join(THREE_TO_ONE.get(r, "X") for r in z["resname"])
                ref = seqs.get(pid)
                if ref is None:
                    bad.append((pid, "in a .list but not in any split JSON")); continue
                if seq != ref[1]:
                    bad.append((pid, f"sequence mismatch vs {ref[0]}.json")); continue
                a = z[args.key]
                if a.ndim != 2 or a.shape[0] != len(seq):
                    bad.append((pid, f"shape {a.shape}")); continue
                if not np.isfinite(a).all():
                    bad.append((pid, "non-finite")); continue
                with zf.open(pid + ".npy", "w", force_zip64=True) as fh:
                    np.lib.format.write_array(fh, np.ascontiguousarray(a, dtype=np.float32))
                done.add(pid)
                dim = a.shape[1]
                n += 1
        print(f"{os.path.basename(tpath)}: {n} structures packed", flush=True)
    zf.close()

    print(f"\npacked {len(done)} structures, key={args.key}, dim={dim if done else '-'}")
    print(f"skipped {dropped} structures not in the current PANAV split lists")
    for split, ids in wanted.items():
        miss = sorted(ids - done)
        print(f"  {split}: {len(ids) - len(miss)}/{len(ids)} present"
              + (f"  missing e.g. {miss[:5]}" if miss else ""))
    if bad:
        print(f"\n{len(bad)} structures failed QC (skipped):")
        for pid, why in bad[:20]:
            print(f"  {pid}: {why}")
    print(f"wrote {out_path} ({os.path.getsize(out_path) / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
