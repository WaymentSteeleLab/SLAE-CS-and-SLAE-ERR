#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NESTED train_{10,20,...,90}.list for the train-size sweep.

One fixed random.Random(1) shuffle of the full train.list, prefixes at each 10% step
-> train_10 ⊂ train_20 ⊂ ... ⊂ train_90 ⊂ train.list. Writes into --out (default: ./lists_v3/).

    # PANAV (also used by LACS / NOREF: same 4647 train ids)
    python3 make_train_size_lists_v3.py --train ../PANAV/train.list --out lists_v3
    # NOREF_whole (its own 12054-id train.list)
    python3 make_train_size_lists_v3.py --train ../NOREF_whole/train.list \
        --out ../NOREF_whole_train_size_sweep/lists_v3

Refuses to overwrite an existing train_*.list in --out. Prints sizes and verifies nesting.
"""
import argparse
import os
import random
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--train", default="../PANAV/train.list")
ap.add_argument("--out", default="lists_v3")
ap.add_argument("--seed", type=int, default=1)
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
