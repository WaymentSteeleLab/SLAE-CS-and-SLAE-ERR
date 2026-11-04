#!/usr/bin/env python
"""Collect the inputs of merge_oof.py from the runs of docs/reproduce.md (steps 1, 2a and 2b).

    python scripts/kfold/collect_runs.py --runs runs --oof oof --out lists

Reads, relative to the directory it is started from:
  <runs>/panav_seed<s>/best_ckpts/top3_val_N_epoch_RMSE_*.ckpt       the 10 full-data models (s = 1..10)
  <runs>/kfold<f>_seed<s>/best_ckpts/top3_val_N_epoch_RMSE_*.ckpt    the 50 K-fold models (f = 0..4)
  <oof>/fold<f>/predictions_N_panav_kfold<f>_seed<s>.json         their out-of-fold predictions
Writes:
  <out>/kfold_ckpts.csv   'fold,seed,ckpt' rows        (--kfold_ckpt_list)
  <out>/full_ckpts.csv    'panav_seed<s>_full,ckpt'    (--val_ckpt_list and --test_ckpt_list; the 2b
                          predictions must use method=panav_seed<s>_full_val / _test)
  <oof>/all/              links to the 50 out-of-fold prediction files (--kfold_pred_dir)
Stops if a checkpoint or prediction is missing, or a run has more than one best-RMSE checkpoint.
"""
import argparse
import os
from pathlib import Path

SEEDS = range(1, 11)
FOLDS = range(5)


def best_ckpt(run):
    found = sorted((run / "best_ckpts").glob("top3_val_N_epoch_RMSE_*.ckpt"))
    if len(found) != 1:
        raise SystemExit(f"{run}: expected one best-RMSE checkpoint in best_ckpts/, found {len(found)}")
    return found[0].resolve()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--oof", type=Path, default=Path("oof"))
    ap.add_argument("--out", type=Path, default=Path("lists"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    kfold = [(f, s, best_ckpt(a.runs / f"kfold{f}_seed{s}")) for f in FOLDS for s in SEEDS]
    full = [(f"panav_seed{s}_full", best_ckpt(a.runs / f"panav_seed{s}")) for s in SEEDS]
    (a.out / "kfold_ckpts.csv").write_text("".join(f"{f},{s},{c}\n" for f, s, c in kfold))
    (a.out / "full_ckpts.csv").write_text("".join(f"{t},{c}\n" for t, c in full))

    link_dir = a.oof / "all"
    link_dir.mkdir(parents=True, exist_ok=True)
    for f in FOLDS:
        for s in SEEDS:
            name = f"predictions_N_panav_kfold{f}_seed{s}.json"
            src = (a.oof / f"fold{f}" / name).resolve()
            if not src.exists():
                raise SystemExit(f"missing out-of-fold prediction {src}")
            dst = link_dir / name
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            os.symlink(src, dst)
    print(f"{len(kfold)} K-fold and {len(full)} full-data checkpoints -> {a.out}/kfold_ckpts.csv, "
          f"{a.out}/full_ckpts.csv; {len(kfold)} out-of-fold predictions linked in {link_dir}/")


if __name__ == "__main__":
    main()
