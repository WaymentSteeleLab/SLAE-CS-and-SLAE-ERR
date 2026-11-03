#!/usr/bin/env python
"""Pre-flight check of input_structures/ before running the SLAE-CS predictors on it (read-only).

Reports, per *.pdb: protein id (file name up to the first "."), number of residues with a CA,
chains, non-standard residues, and problems that would silently lose a protein in the job:
  * two files mapping to the same id (slae_cs/inference/dataset.py uses pdb.split('.')[0])
  * unparsable / empty files, no ATOM records, no CA atoms
  * very long proteins (GPU memory), multiple chains (the model was trained on single entities)

    python3 check_inputs.py [--dir input_structures]
"""
import argparse
import collections
import os
import sys

STD = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET",
       "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"}
ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "input_structures"))
ap.add_argument("--long", type=int, default=600, help="warn above this many residues")
a = ap.parse_args()

files = sorted(f for f in os.listdir(a.dir) if f.endswith(".pdb"))
others = sorted(f for f in os.listdir(a.dir) if not f.endswith(".pdb"))
if not files:
    sys.exit(f"no *.pdb files in {a.dir}")
ids = collections.defaultdict(list)
for f in files:
    ids[f.split(".")[0]].append(f)

errors, warns = [], []
for pid, fs in ids.items():
    if len(fs) > 1:
        errors.append(f"{pid}: several files map to the same id: {fs}")
print(f"{'id':<28}{'residues':>9}  chains  non-standard")
for f in files:
    pid = f.split(".")[0]
    res, chains, nonstd = [], set(), set()
    try:
        for line in open(os.path.join(a.dir, f), errors="replace"):
            if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA":
                key = (line[21], line[22:27])
                if not res or res[-1] != key:
                    res.append(key)
                chains.add(line[21])
                if line[17:20] not in STD:
                    nonstd.add(line[17:20])
            if line.startswith("ENDMDL"):
                break  # first model only
    except OSError as e:
        errors.append(f"{f}: {e}")
        continue
    if not res:
        errors.append(f"{f}: no CA atoms")
        continue
    if len(chains) > 1:
        warns.append(f"{pid}: {len(chains)} chains {sorted(chains)} -- predicted as one input (model trained on single entities)")
    if len(res) > a.long:
        warns.append(f"{pid}: {len(res)} residues -- may need a GPU with more memory")
    if " " in f:
        errors.append(f"{f}: space in file name")
    print(f"{pid:<28}{len(res):>9}  {''.join(sorted(chains)):<6}  {','.join(sorted(nonstd)) or '-'}")
if others:
    warns.append(f"not used as input (only *.pdb are sent to the predictor): {others[:5]}{' ...' if len(others) > 5 else ''}")
print(f"\n{len(files)} files, {len(ids)} protein ids")
for w in warns:
    print("WARNING " + w)
for e in errors:
    print("ERROR   " + e)
sys.exit(1 if errors else 0)
