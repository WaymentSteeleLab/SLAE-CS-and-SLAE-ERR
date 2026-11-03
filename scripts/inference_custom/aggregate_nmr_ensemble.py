#!/usr/bin/env python
"""Average the 10-seed predictions over the models of each NMR ensemble.

Input : results/ensemble_N_10seed_custom.json (make_ensemble_custom.py; one entry per single-model
        file <protein>_mXX, pred [10 seeds][L]) and input_structures/nmr_models.tsv
        (split_nmr_models.py: file_id -> protein, model).
Per protein and residue, over all M models x 10 seeds = 10M predictions:
    N_mean     mean over models of the per-model 10-seed mean (= mean of all 10M predictions)
    std_seed   RMS over models of the per-model seed std (ddof=1)   -> model (epistemic) spread
    std_model  std over models (ddof=1) of the per-model 10-seed mean -> conformational spread
    std_total  std over all 10M predictions (ddof=1)
A residue is kept only if every model's mask keeps it. All models of a protein must have the same
length and (per nmr_models.tsv) the same residue list, otherwise the protein is skipped with a message.
Output: results/nmr_ensemble_N.json  {protein: {models, mask, N_mean, std_seed, std_model, std_total}}
        results/nmr_ensemble_N.csv   protein, index, chain, resnum, resname, n_models, N_mean, std_seed, std_model, std_total

    python3 aggregate_nmr_ensemble.py
"""
import argparse
import collections
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--ensemble", default=os.path.join(HERE, "results", "ensemble_N_10seed_custom.json"))
ap.add_argument("--manifest", default=os.path.join(HERE, "input_structures", "nmr_models.tsv"))
ap.add_argument("--structures", default=os.path.join(HERE, "input_structures"))
ap.add_argument("--out", default=os.path.join(HERE, "results", "nmr_ensemble_N"))
a = ap.parse_args()

E = json.load(open(a.ensemble))
rows = [l.rstrip("\n").split("\t") for l in open(a.manifest)][1:]
groups = collections.defaultdict(list)
for fid, pid, k, n, ok in rows:
    groups[pid].append((int(k), fid, ok == "1"))


def ca_residues(fid):
    out = []
    for l in open(os.path.join(a.structures, fid + ".pdb"), errors="replace"):
        if l.startswith(("ATOM", "HETATM")) and l[12:16].strip() == "CA":
            key = (l[21], l[22:27].strip(), l[17:20])
            if not out or out[-1] != key:
                out.append(key)
    return out


out_json, out_rows, msgs = {}, [], []
for pid, ms in sorted(groups.items()):
    ms.sort()
    missing = [fid for _, fid, _ in ms if fid not in E["pred"]]
    if missing:
        msgs.append(f"{pid}: no prediction for {missing[:3]} (featuriser skip?) -- using the remaining {len(ms) - len(missing)} model(s)")
        ms = [m for m in ms if m[1] in E["pred"]]
    if not ms:
        continue
    if not all(ok for _, _, ok in ms):
        msgs.append(f"{pid}: models differ in residue list -- skipped"); continue
    P = [np.array(E["pred"][fid], dtype=np.float64) for _, fid, _ in ms]   # each [10, L]
    if len({p.shape for p in P}) != 1:
        msgs.append(f"{pid}: prediction lengths differ between models -- skipped"); continue
    P = np.stack(P)                                                        # [M, 10, L]
    mask = np.all([E["mask"][fid] for _, fid, _ in ms], axis=0)
    M = P.shape[0]
    per_model_mean = P.mean(1)                                             # [M, L]
    N_mean = per_model_mean.mean(0)
    std_seed = np.sqrt((P.std(1, ddof=1) ** 2).mean(0))
    std_model = per_model_mean.std(0, ddof=1) if M > 1 else np.zeros_like(N_mean)
    std_total = P.reshape(-1, P.shape[2]).std(0, ddof=1)
    nan = lambda v: [None if not m else float(x) for x, m in zip(v, mask)]
    out_json[pid] = {"models": [k for k, _, _ in ms], "mask": mask.tolist(), "N_mean": nan(N_mean),
                     "std_seed": nan(std_seed), "std_model": nan(std_model), "std_total": nan(std_total)}
    res = ca_residues(ms[0][1])
    if len(res) != P.shape[2]:
        msgs.append(f"{pid}: CA count {len(res)} != prediction length {P.shape[2]} -- chain/resnum left empty")
        res = [("", "", "")] * P.shape[2]
    for i in range(P.shape[2]):
        if mask[i]:
            out_rows.append([pid, i, *res[i], M, f"{N_mean[i]:.4f}", f"{std_seed[i]:.4f}", f"{std_model[i]:.4f}", f"{std_total[i]:.4f}"])

json.dump(out_json, open(a.out + ".json", "w"))
with open(a.out + ".csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["protein", "index", "chain", "resnum", "resname", "n_models", "N_mean", "std_seed", "std_model", "std_total"])
    w.writerows(out_rows)
print(f"{len(out_json)} protein(s), {len(out_rows)} residues -> {a.out}.json / .csv")
for m in msgs:
    print("NOTE " + m)
