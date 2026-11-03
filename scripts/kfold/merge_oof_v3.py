#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build the error head's per-residue input table (oof_json) from the SLAE-CS predictions:

  train : K-fold OOF -- 5 folds x seeds 1-10 (folds: splits/kfold/, cluster-respecting);
          each train id predicted by the 10 models of the fold that held it out (tag mask only).
  val   : the 10 full-data PANAV models on val (flags + tag mask).
  test  : the same 10 models on test (flags + tag mask).
Inputs are given on the command line (see docs/reproduce.md).

Per residue it writes the 10-model ensemble mean ("N"), the AND of the 10 masks ("N_mask"),
the std over models (ddof=1, "N_std") and the 10 predictions sorted ascending and centered on
the mean ("N_sort00".."N_sort09" = sorted value - mean). Keys are sequential strings, schema is
a drop-in for cs_oof_data.build_oof_resid_dict / build_oof_std_dict / build_oof_sorted_dict.

Refuses to write anything if any check fails:
  * every expected prediction file exists and is NEWER than its checkpoint (no stale outputs);
  * each fold's predicted ids == heldout_fold{k}.list, val/test ids cover PANAV val/test;
  * every prediction has exactly the sequence length (batch-1 inference, no padding);
  * every train/val/test protein in the PANAV jsons gets an entry.

    python3 scripts/kfold/merge_oof_v3.py --data_dir data/cs --folds_dir splits/kfold \
        --kfold_pred_dir <dir> --kfold_ckpt_list <fold,seed,ckpt list> \
        --val_pred_dir <dir> --val_ckpt_list <run_tag,ckpt list> \
        --test_pred_dir <dir> --test_ckpt_list <run_tag,ckpt list> \
        --out data/cs/train_val_test_oof_ensemble_10seed_v3_N.json
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]                  # repository root (error_head/cs_oof_data.py)
# set from the command line in main()
PANAV = KFOLD = KFOLD_PRED_DIR = KFOLD_CKPT_LIST = OUT = None
VAL_PRED_DIR = VAL_CKPT_LIST = TEST_PRED_DIR = TEST_CKPT_LIST = None
SEEDS = list(range(1, 11))
N_FOLDS = 5
SORTED_FIELDS = [f"N_sort{i:02d}" for i in range(len(SEEDS))]

problems = []


def fail(msg):
    problems.append(msg)


def pid_of(eid):
    return eid.lower().removeprefix("bmr")


def load_split(name):
    d = json.load(open(PANAV / f"{name}.json"))
    return d, {pid_of(d["entry_ID"][k]): k for k in d["entry_ID"]}


def read_list(p):
    return [l.strip() for l in open(p) if l.strip()]


def load_pred(path, ckpt):
    if not os.path.exists(path):
        fail(f"missing prediction file {path}")
        return None
    if os.path.getmtime(path) <= os.path.getmtime(ckpt):
        fail(f"prediction older than its checkpoint (stale?): {path}")
    d = json.load(open(path))
    return d["pred"], d["mask"]


def stats(preds, masks):
    """preds/masks: M arrays of equal length n -> mean, std (ddof=1), sorted-centred [M, n], AND-mask."""
    P = np.stack(preds).astype(np.float64)
    m = np.logical_and.reduce(np.stack(masks).astype(bool))
    mean = np.zeros(P.shape[1]); std = np.zeros(P.shape[1]); srt = np.zeros_like(P)
    if m.any():
        mean[m] = P[:, m].mean(0)
        std[m] = P[:, m].std(0, ddof=1)
        srt[:, m] = np.sort(P[:, m], axis=0) - mean[m]
    return mean, std, srt, m


def parse_args():
    ap = argparse.ArgumentParser(description="Build the SLAE-ERR input table from the SLAE-CS predictions.")
    ap.add_argument("--data_dir", required=True, help="PANAV {train,val,test}.json and tag_labels.csv (data/cs)")
    ap.add_argument("--folds_dir", required=True, help="heldout_fold{k}.list (splits/kfold)")
    ap.add_argument("--kfold_pred_dir", required=True, help="predictions_N_panav_v3_kfold{f}_seed{s}.json")
    ap.add_argument("--kfold_ckpt_list", required=True, help="'fold,seed,ckpt_path' rows of the K-fold models")
    ap.add_argument("--val_pred_dir", required=True, help="predictions_N_<run_tag>_val.json of the 10 models")
    ap.add_argument("--val_ckpt_list", required=True, help="'run_tag,ckpt_path' rows of the 10 models (val)")
    ap.add_argument("--test_pred_dir", required=True, help="predictions_N_<run_tag>_test.json of the 10 models")
    ap.add_argument("--test_ckpt_list", required=True, help="'run_tag,ckpt_path' rows of the 10 models (test)")
    ap.add_argument("--out", required=True, help="output json (a _provenance.tsv is written next to it)")
    return ap.parse_args()


def main():
    global PANAV, KFOLD, KFOLD_PRED_DIR, KFOLD_CKPT_LIST, OUT
    global VAL_PRED_DIR, VAL_CKPT_LIST, TEST_PRED_DIR, TEST_CKPT_LIST
    a = parse_args()
    PANAV, KFOLD, OUT = Path(a.data_dir), Path(a.folds_dir), Path(a.out)
    KFOLD_PRED_DIR, KFOLD_CKPT_LIST = Path(a.kfold_pred_dir), Path(a.kfold_ckpt_list)
    VAL_PRED_DIR, VAL_CKPT_LIST = Path(a.val_pred_dir), Path(a.val_ckpt_list)
    TEST_PRED_DIR, TEST_CKPT_LIST = Path(a.test_pred_dir), Path(a.test_ckpt_list)
    splits = {s: load_split(s) for s in ("train", "val", "test")}
    out = {"N": {}, "N_mask": {}, "N_std": {}, "entry_ID": {}, "sequence": {}, **{f: {} for f in SORTED_FIELDS}}
    prov = []

    def emit(split, pid, mean, std, srt, mask):
        d, idx = splits[split]
        k = idx[pid]
        seq = d["sequence"][k]
        if len(mean) != len(seq):
            fail(f"{split} {pid}: prediction length {len(mean)} != sequence length {len(seq)}")
            return
        key = str(len(prov))
        out["N"][key] = mean.tolist(); out["N_mask"][key] = mask.tolist(); out["N_std"][key] = std.tolist()
        for i, f in enumerate(SORTED_FIELDS):
            out[f][key] = srt[i].tolist()
        out["entry_ID"][key] = d["entry_ID"][k]; out["sequence"][key] = seq
        prov.append((key, pid, split))

    # ---------------- train: K-fold OOF ---------------- #
    ck = {}
    for l in read_list(KFOLD_CKPT_LIST):
        f, s, c = l.split(",")
        ck[(int(f), int(s))] = c
    if len(ck) != N_FOLDS * len(SEEDS):
        fail(f"{KFOLD_CKPT_LIST} has {len(ck)} (fold, seed) rows, expected {N_FOLDS * len(SEEDS)}")
    seen_train = set()
    for f in range(N_FOLDS):
        held = [x.lower() for x in read_list(KFOLD / f"heldout_fold{f}.list")]
        srcs = []
        for s in SEEDS:
            r = load_pred(KFOLD_PRED_DIR / f"predictions_N_panav_v3_kfold{f}_seed{s}.json", ck.get((f, s), "/dev/null"))
            if r is None:
                continue
            if {k.lower() for k in r[0]} != set(held):
                fail(f"fold {f} seed {s}: predicted ids != heldout_fold{f}.list")
            srcs.append(r)
        if len(srcs) != len(SEEDS):
            continue
        for pid in held:
            if pid not in splits["train"][1]:
                fail(f"heldout id {pid} not in PANAV/train.json"); continue
            if pid in seen_train:
                fail(f"train id {pid} held out in more than one fold")
            seen_train.add(pid)
            emit("train", pid, *stats([np.array(p[pid]) for p, _ in srcs], [np.array(m[pid]) for _, m in srcs]))

    # ---------------- val / test: full-data 10-seed models ---------------- #
    for split, d, ckfile in (("val", VAL_PRED_DIR, VAL_CKPT_LIST),
                             ("test", TEST_PRED_DIR, TEST_CKPT_LIST)):
        rows = [l.split(",") for l in read_list(ckfile)]
        seeds = sorted(int(re.search(r"_seed(\d+)_", t).group(1)) for t, _ in rows)
        if seeds != SEEDS:
            fail(f"{ckfile}: seeds {seeds} != 1..10")
        srcs = [load_pred(d / f"predictions_N_{t}_{split}.json", c) for t, c in rows]
        srcs = [s for s in srcs if s is not None]
        if len(srcs) != len(SEEDS):
            continue
        for pid in splits[split][1]:
            if any(pid not in p for p, _ in srcs):
                fail(f"{split} {pid}: missing from some model's predictions"); continue
            emit(split, pid, *stats([np.array(p[pid]) for p, _ in srcs], [np.array(m[pid]) for _, m in srcs]))

    covered = {sp: {p for _, p, s in prov if s == sp} for sp in ("train", "val", "test")}
    for sp in ("train", "val", "test"):
        missing = set(splits[sp][1]) - covered[sp]
        if missing:
            fail(f"{sp}: {len(missing)} proteins have no entry, e.g. {sorted(missing)[:5]}")

    if problems:
        print(f"NOT writing {OUT.name} -- {len(problems)} problem(s):")
        for p in problems[:30]:
            print("  " + p)
        sys.exit(1)

    json.dump(out, open(OUT, "w"))
    with open(str(OUT).replace(".json", "_provenance.tsv"), "w") as f:
        f.write("out_key\tpdb_id\tsplit\n")
        f.writelines(f"{k}\t{p}\t{s}\n" for k, p, s in prov)
    print(f"wrote {OUT}  ({len(prov)} proteins: " +
          ", ".join(f"{sp} {len(covered[sp])}" for sp in ("train", "val", "test")) + ")")

    # residual summary on the same masks the error head uses (+ label validity + tag mask)
    sys.path.insert(0, str(REPO / "error_head"))
    from cs_oof_data import build_oof_resid_dict
    for sp in ("train", "val", "test"):
        r = build_oof_resid_dict(OUT, **{f"{sp}_json_path": PANAV / f"{sp}.json"}, atom="N",
                                 tag_mask_csv=PANAV / "tag_labels.csv")
        v = np.concatenate([x[0][x[1]].numpy() for x in r.values() if x[1].any()])
        print(f"  {sp:5s}: residues={len(v):6d}  mean|resid|={np.abs(v).mean():.4f}  RMSE={np.sqrt((v ** 2).mean()):.4f}")
    print("  (the error head initialises its output bias to the TRAIN mean|resid| at run time)")


if __name__ == "__main__":
    main()
