#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NESTED train_{10,20,...,90}.list for the train-size sweep.

One fixed random.Random(1) shuffle of the full train.list, prefixes at each 10% step
-> train_10 ⊂ train_20 ⊂ ... ⊂ train_90 ⊂ train.list. Writes into --out. Run from the repository root;
the outputs are identical to splits/train_size_sweep/.

    # rBMRB (PANAV; also used by LACS and BASELINE: the same 4647 training ids)
    python scripts/train_size_sweep/make_train_size_lists.py --train splits/panav/train.list --out <dir>
    # whole BMRB (its own 12054 training ids)
    python scripts/train_size_sweep/make_train_size_lists.py --train splits/whole_bmrb/train.list --out <dir>

Refuses to overwrite an existing train_*.list in --out. Prints sizes and verifies nesting.
"""
import argparse
import os
import random
import sys

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--train", required=True, help="full training id list, e.g. splits/panav/train.list")
ap.add_argument("--out", required=True, help="output directory for train_{10..90}.list")
ap.add_argument("--seed", type=int, default=1, help="shuffle seed (1 for the lists of the paper)")
args = ap.parse_args()

with open(args.train) as f:
    ids = [line.strip() for line in f if line.strip()]
if len(ids) != len(set(ids)):
    sys.exit(f"{args.train}: duplicate ids")
n = len(ids)
print(f"{args.train}: {n} train ids, seed {args.seed} -> {args.out}/")

os.makedirs(args.out, exist_ok=True)
existing = [p for p in os.listdir(args.out) if p.startswith("train_") and p.endswith(".list")]
if existing:
    sys.exit(f"refusing to overwrite existing lists in {args.out}: {sorted(existing)}")

rng = random.Random(args.seed)
shuffled = ids[:]
rng.shuffle(shuffled)

prev = set()
for pct in range(10, 100, 10):
    k = round(n * pct / 100)
    subset = shuffled[:k]
    assert prev <= set(subset), "nesting broken"
    prev = set(subset)
    with open(os.path.join(args.out, f"train_{pct}.list"), "w") as f:
        f.write("\n".join(subset) + "\n")
    print(f"  {pct:2d}%  {k:6d} ids  train_{pct}.list")
print("nested: train_10 ⊂ ... ⊂ train_90 ⊂ full train.list -- OK")
