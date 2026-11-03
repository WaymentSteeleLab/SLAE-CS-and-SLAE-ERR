#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Merge the per-seed chemical_shift_predictor.py outputs (predictions_{atom}_{tag}.json,
{"pred": {pdb_id: [...]}, "mask": {pdb_id: [...]}}) into one json keyed by pdb_id: the per-seed
predictions stacked, the mask, and the ensemble mean and std (ddof=1) over the masked residues.

The mask depends only on the structure and the flags / tag files, not on the weights, so it must
be identical across seeds; the script checks this and warns (it does not AND the masks) if not.

Run:
    python merge_seed_predictions.py --pred <predictions of seeds 1..10> \
        --seeds 1 2 3 4 5 6 7 8 9 10 --out predictions_N_seed_ensemble.json
"""
import argparse
import json

import numpy as np


def load(path):
    with open(path) as f:
        d = json.load(f)
    return d["pred"], d["mask"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", nargs="+", required=True,
                     help="predictions_{atom}_{tag}.json files, one per seed, same order as --seeds")
    ap.add_argument("--seeds", nargs="+", required=True, type=int)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert len(args.pred) == len(args.seeds), "--pred and --seeds must be the same length"

    per_seed_pred = {}   # seed -> {pdb_id: [..]}
    per_seed_mask = {}   # seed -> {pdb_id: [..]}
    for seed, path in zip(args.seeds, args.pred):
        pred, mask = load(path)
        per_seed_pred[seed] = pred
        per_seed_mask[seed] = mask

    # ids present in every seed's output
    id_sets = [set(per_seed_pred[s].keys()) for s in args.seeds]
    common_ids = set.intersection(*id_sets)
    dropped = set.union(*id_sets) - common_ids
    if dropped:
        print(f"WARNING: {len(dropped)} pdb_id(s) missing from at least one seed's "
              f"output, excluded from the merge: {sorted(dropped)[:10]}"
              f"{'...' if len(dropped) > 10 else ''}")

    out = {"seeds": args.seeds, "pred": {}, "mask": {}, "mean": {}, "std": {}}
    mask_mismatches = []

    for pdb_id in sorted(common_ids, key=lambda x: str(x)):
        preds = [per_seed_pred[s][pdb_id] for s in args.seeds]
        masks = [per_seed_mask[s][pdb_id] for s in args.seeds]

        n = min(len(p) for p in preds + masks)
        preds = np.array([p[:n] for p in preds], dtype=np.float64)   # [n_seeds, n_res]
        masks = np.array([m[:n] for m in masks], dtype=bool)          # [n_seeds, n_res]

        if not np.all(masks == masks[0]):
            mask_mismatches.append(pdb_id)
        mask = masks[0]  # the first seed's mask; mismatches are reported below

        mean = np.full(n, np.nan)
        std = np.full(n, np.nan)
        if mask.any():
            mean[mask] = preds[:, mask].mean(axis=0)
            std[mask] = preds[:, mask].std(axis=0, ddof=1) if len(args.seeds) > 1 else 0.0

        out["pred"][pdb_id] = preds.tolist()          # [n_seeds, n_res]
        out["mask"][pdb_id] = mask.tolist()
        out["mean"][pdb_id] = np.where(np.isnan(mean), None, mean).tolist()
        out["std"][pdb_id] = np.where(np.isnan(std), None, std).tolist()

    if mask_mismatches:
        print(f"WARNING: mask differed across seeds for {len(mask_mismatches)} pdb_id(s) "
              f"-- this should not happen if all seeds were run against the same "
              f"structures/mask_json/tag_mask_csv. Affected ids: {mask_mismatches[:10]}"
              f"{'...' if len(mask_mismatches) > 10 else ''}")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {len(common_ids)} ids ({len(args.seeds)} seeds each) to {args.out}")


if __name__ == "__main__":
    main()
