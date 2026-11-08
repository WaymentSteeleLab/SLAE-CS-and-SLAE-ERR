#!/usr/bin/env python
"""Summarise embedding-benchmark runs from their training logs (stdout).

Only checkpoints are kept from each run, so the metrics come from the summary
block embedding_shift_trainer.py prints at the end of each run
("=== summary ===" followed by JSON). If a run was resubmitted, the newest
log that contains a summary wins.

LR sweep : one row per LR (for the given --seed), sorted by best val RMSE;
           the LR with the lowest val RMSE is the one to use for the seeds.
Seeds    : mean +- sd over seeds at a fixed LR (test metrics are taken at each
           run's best-val epoch, never selected on test).

    python3 collect_results.py --emb <emb> --atom N --seed 42        # LR sweep
    python3 collect_results.py --emb <emb> --atom N --lr <lr>         # seeds
"""
import argparse
import glob
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")
SEED_LIST = os.path.join(HERE, "seed_list.txt")
MARKER = "=== summary ==="
METRICS = ("RMSE", "MAE", "Pearson", "Spearman")


def read_summary(path):
    with open(path, errors="replace") as f:
        text = f.read()
    i = text.rfind(MARKER)
    if i < 0:
        return None
    try:
        s, _ = json.JSONDecoder().raw_decode(text[i + len(MARKER):].lstrip())
    except ValueError:
        return None
    return s if "best_epoch" in s else None


def load(log_dir, emb, atom):
    """run_name -> summary, newest log first."""
    runs = {}
    logs = glob.glob(os.path.join(log_dir, f"*_{emb}_{atom}_lr*_seed*.out"))
    for p in sorted(logs, key=os.path.getmtime, reverse=True):
        s = read_summary(p)
        if s is None or s["run_name"] in runs:
            continue
        s["lr_str"] = s["run_name"].split("_lr")[1].split("_seed")[0]
        runs[s["run_name"]] = s
    return list(runs.values())


def split_sizes(r):
    return f"{r['n_train']}/{r['n_val']}/{r['n_test']}"


def warn_mixed_splits(rows):
    """Warn if the runs were trained on different split sizes: such runs must not be compared
    or averaged together."""
    sizes = {split_sizes(r) for r in rows}
    if len(sizes) > 1:
        print(f"\n!! runs use different PANAV splits {sorted(sizes)} -- "
              "resubmit the stale ones before comparing")


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--log_dir", default=LOG_DIR)
    a.add_argument("--emb", required=True)
    a.add_argument("--atom", default="N")
    g = a.add_mutually_exclusive_group(required=True)
    g.add_argument("--seed", type=int, help="LR-sweep view for this seed")
    g.add_argument("--lr", help="seed view for this LR (as written in the run name, e.g. 1e-4)")
    a.add_argument("--seed_list", default=SEED_LIST,
                   help="seed view only counts these seeds (keeps the LR-sweep seed out)")
    args = a.parse_args()

    runs = load(args.log_dir, args.emb, args.atom)
    if args.seed is not None:
        rows = sorted((r for r in runs if r["seed"] == args.seed), key=lambda r: r["val_RMSE"])
        print(f"LR sweep  {args.emb}  atom={args.atom}  seed={args.seed}  ({len(rows)} finished runs)")
        print(f"{'lr':>8} {'epoch':>5} {'val_RMSE':>9} {'val_Spear':>9} {'test_RMSE':>9} {'test_Spear':>10}  n_train/val/test")
        for r in rows:
            print(f"{r['lr_str']:>8} {r['best_epoch']:>5} {r['val_RMSE']:>9.4f} {r['val_Spearman']:>9.4f}"
                  f" {r['test_RMSE']:>9.4f} {r['test_Spearman']:>10.4f}  {split_sizes(r)}")
        warn_mixed_splits(rows)
        if rows:
            edges = {min(rows, key=lambda r: r["lr"])["lr_str"], max(rows, key=lambda r: r["lr"])["lr_str"]}
            print(f"\n-> best LR by val RMSE: {rows[0]['lr_str']}"
                  + ("   (at the edge of the grid -- consider extending it)" if rows[0]["lr_str"] in edges else ""))
    else:
        with open(args.seed_list) as f:
            seeds = {int(l) for l in f if l.strip()}
        rows = sorted((r for r in runs if r["lr_str"] == args.lr and r["seed"] in seeds),
                      key=lambda r: r["seed"])
        print(f"Seeds  {args.emb}  atom={args.atom}  lr={args.lr}  ({len(rows)} finished runs: "
              f"seeds {[r['seed'] for r in rows]})")
        for split in ("val", "test"):
            parts = []
            for m in METRICS:
                v = [r[f"{split}_{m}"] for r in rows]
                sd = st.stdev(v) if len(v) > 1 else 0.0
                parts.append(f"{m} {st.mean(v):.4f} +- {sd:.4f}")
            print(f"  {split:>4}: " + "   ".join(parts))
        print(f"  best epochs: {[r['best_epoch'] for r in rows]}")
        print(f"  n_train/val/test: {sorted({split_sizes(r) for r in rows})}")
        warn_mixed_splits(rows)


if __name__ == "__main__":
    main()
