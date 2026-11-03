#!/usr/bin/env python
"""Split multi-MODEL PDB files (NMR ensembles) into one PDB per model for the custom inference.

slae_cs/inference/dataset.py only ever uses the FIRST model of a multi-model file ("has multiple models, using the
first one"), so an NMR ensemble has to be split before inference to get a prediction per model.

For every <name>.pdb in --src this writes, into --out:
    <id>_m01.pdb, <id>_m02.pdb, ...   (id = <name> with "." replaced by "_"; a file without
                                       MODEL records becomes a single <id>_m01.pdb)
each holding that model's ATOM/HETATM/TER records + END, and a manifest
    nmr_models.tsv : file_id, protein, model, n_CA_residues, sequence_ok
It checks that all models of one protein have the same CA residue list (chain, resnum, resname) --
required to average predictions over models -- and refuses to overwrite existing files in --out.

    python3 split_nmr_models.py --src nmr_raw --out input_structures
"""
import argparse
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="directory with the original (multi-model) *.pdb files")
ap.add_argument("--out", default="input_structures")
a = ap.parse_args()

srcs = sorted(f for f in os.listdir(a.src) if f.endswith(".pdb"))
if not srcs:
    sys.exit(f"no *.pdb in {a.src}")
os.makedirs(a.out, exist_ok=True)


def models_of(path):
    """list of models, each a list of ATOM/HETATM/TER lines (whole file = one model if no MODEL)."""
    models, cur, in_model, saw_model = [], [], False, False
    for line in open(path, errors="replace"):
        rec = line[:6]
        if rec.startswith("MODEL"):
            saw_model, in_model, cur = True, True, []
        elif rec.startswith("ENDMDL"):
            if cur:
                models.append(cur)
            in_model, cur = False, []
        elif rec.startswith(("ATOM", "HETATM", "TER")):
            if in_model or not saw_model:
                cur.append(line if line.endswith("\n") else line + "\n")
    if cur and (in_model or not saw_model):
        # no MODEL records at all, or the last MODEL was never closed by ENDMDL
        models.append(cur)
    return models


def ca_residues(lines):
    out = []
    for l in lines:
        if l.startswith(("ATOM", "HETATM")) and l[12:16].strip() == "CA":
            key = (l[21], l[22:27].strip(), l[17:20])
            if not out or out[-1] != key:
                out.append(key)
    return out


plan, problems = [], []
for f in srcs:
    pid = f[:-4].replace(".", "_").replace(" ", "_")
    ms = models_of(os.path.join(a.src, f))
    if not ms:
        problems.append(f"{f}: no ATOM records")
        continue
    ref = ca_residues(ms[0])
    for k, lines in enumerate(ms, 1):
        res = ca_residues(lines)
        fid = f"{pid}_m{k:02d}"
        if os.path.exists(os.path.join(a.out, fid + ".pdb")):
            problems.append(f"{fid}.pdb already exists in {a.out} (refusing to overwrite)")
        plan.append((fid, pid, k, lines, len(res), res == ref))
    print(f"{f}: {len(ms)} model(s), {len(ref)} CA residues" +
          ("" if all(p[5] for p in plan if p[1] == pid) else "  ** residue lists differ between models **"))
if problems:
    sys.exit("NOT writing anything:\n  " + "\n  ".join(problems))

for fid, pid, k, lines, n, ok in plan:
    with open(os.path.join(a.out, fid + ".pdb"), "w") as fh:
        fh.writelines(lines)
        fh.write("END\n")
with open(os.path.join(a.out, "nmr_models.tsv"), "w") as fh:
    fh.write("file_id\tprotein\tmodel\tn_CA_residues\tsequence_ok\n")
    fh.writelines(f"{fid}\t{pid}\t{k}\t{n}\t{int(ok)}\n" for fid, pid, k, _, n, ok in plan)
bad = sorted({p[1] for p in plan if not p[5]})
print(f"\nwrote {len(plan)} single-model PDBs for {len({p[1] for p in plan})} protein(s) -> {a.out}/ (+ nmr_models.tsv)")
if bad:
    print(f"WARNING models with a different residue list than model 1 (cannot be averaged per residue): {bad}")
