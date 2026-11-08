#!/usr/bin/env python
"""Summarise ProteinMPNN fine-tuning runs from their training logs (stdout).

Fine-tuning goes through the unmodified slae_cs/train/chemical_shift_trainer.py, which
(unlike embedding_shift_trainer.py) prints no summary JSON. Metrics are parsed
from its per-epoch lines instead:
    [epoch 003] split=val epoch_RMSE=... epoch_MAE=... pearson=... spearman=...
The sanity-check pass also prints an "epoch 000" line before training starts,
so the LAST line per (epoch, split) wins. A run counts as finished only once the
trainer printed "=== Copied best checkpoints to ===". If a run was
resubmitted, the newest finished log wins.

Best epoch = lowest val RMSE; test metrics are taken at that same epoch (never
selected on test), same rule as benchmarks/embedding_head/collect_results.py.

    python3 collect_results_ft.py --seed 42                        # (lr, lr_last) sweep
    python3 collect_results_ft.py --lr 1e-4 --lr_last 3e-4         # seeds
"""
import argparse
import glob
import os
import re
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")
SEED_LIST = os.path.join(HERE, "seed_list.txt")
DONE = "=== Copied best checkpoints to ==="
METRICS = ("RMSE", "MAE", "Pearson", "Spearman")
EPOCH_RE = re.compile(r"\[epoch (\d+)\] split=(val|test) epoch_RMSE=(\S+) epoch_MAE=(\S+) "
                      r"pearson=(\S+) spearman=(\S+)")
NAME_RE = re.compile(r"_mpnn_ft_(\w+?)_lr([^_]+)_lrlast([^_]+)_seed(\d+)\.out$")
# chemical_shift_trainer.py prints len(train/val/test ids) right after loading the JSONs
SIZES_RE = re.compile(r"^(\d+)\n(\d+)\n(\d+)\n", re.M)


def read_run(path):
    m = NAME_RE.search(os.path.basename(path))
    if not m:
        return None
    with open(path, errors="replace") as f:
        text = f.read()
    if DONE not in text:
        return None
    ep = {}
    for e, split, *vals in EPOCH_RE.findall(text):
        ep[(int(e), split)] = dict(zip(METRICS, map(float, vals)))
    val = {e: h for (e, s), h in ep.items() if s == "val"}
    if not val:
        return None
    best = min(val, key=lambda e: val[e]["RMSE"])
    sizes = SIZES_RE.search(text)
    r = {"atom": m[1], "lr_str": m[2], "lr_last_str": m[3], "seed": int(m[4]),
         "best_epoch": best, "n_epochs": len(val),
         "sizes": "/".join(sizes.groups()) if sizes else "?"}
    for split in ("val", "test"):
        for k, v in ep.get((best, split), {}).items():
            r[f"{split}_{k}"] = v
    return r


def load(log_dir, atom):
    runs = {}
    logs = glob.glob(os.path.join(log_dir, f"[0-9]*_mpnn_ft_{atom}_lr*_lrlast*_seed*.out"))
    for p in sorted(logs, key=os.path.getmtime, reverse=True):
        r = read_run(p)
        if r is None:
            continue
        key = (r["lr_str"], r["lr_last_str"], r["seed"])
        runs.setdefault(key, r)
    return list(runs.values())


def warn_mixed_splits(rows):
    """Warn if the runs were trained on different split sizes: such runs must not be compared
    or averaged together."""
    sizes = {r["sizes"] for r in rows}
    if len(sizes) > 1:
        print(f"\n!! runs use different PANAV splits {sorted(sizes)} -- "
              "resubmit the stale ones before comparing")


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--log_dir", default=LOG_DIR)
    a.add_argument("--atom", default="N")
    a.add_argument("--seed", type=int, help="sweep view for this seed")
    a.add_argument("--lr", help="seed view: head lr as written in the run name, e.g. 1e-4")
    a.add_argument("--lr_last", help="seed view: encoder lr_last as written in the run name")
    a.add_argument("--seed_list", default=SEED_LIST,
                   help="seed view only counts these seeds (keeps the sweep seed out)")
    args = a.parse_args()
    if (args.seed is None) == (args.lr is None and args.lr_last is None):
        a.error("give either --seed, or --lr and --lr_last")
    if args.seed is None and (args.lr is None or args.lr_last is None):
        a.error("seed view needs both --lr and --lr_last")

    runs = load(args.log_dir, args.atom)
    if args.seed is not None:
        rows = sorted((r for r in runs if r["seed"] == args.seed), key=lambda r: r["val_RMSE"])
        print(f"(lr, lr_last) sweep  mpnn_ft  atom={args.atom}  seed={args.seed}  ({len(rows)} finished runs)")
        print(f"{'lr':>7} {'lr_last':>7} {'epoch':>5} {'val_RMSE':>9} {'val_Spear':>9} "
              f"{'test_RMSE':>9} {'test_Spear':>10}  ids(json) train/val/test")
        for r in rows:
            print(f"{r['lr_str']:>7} {r['lr_last_str']:>7} {r['best_epoch']:>5} {r['val_RMSE']:>9.4f} "
                  f"{r['val_Spearman']:>9.4f} {r['test_RMSE']:>9.4f} {r['test_Spearman']:>10.4f}  {r['sizes']}")
        if rows:
            b = rows[0]
            edge = []
            for k in ("lr_str", "lr_last_str"):
                vals = sorted({r[k] for r in rows}, key=float)
                if len(vals) > 1 and b[k] in (vals[0], vals[-1]):
                    edge.append(k[:-4])
            print(f"\n-> best pair by val RMSE: lr={b['lr_str']} lr_last={b['lr_last_str']}"
                  + (f"   ({'/'.join(edge)} at the edge of the grid -- consider extending it)" if edge else ""))
        warn_mixed_splits(rows)
    else:
        with open(args.seed_list) as f:
            seeds = {int(l) for l in f if l.strip()}
        rows = sorted((r for r in runs if r["lr_str"] == args.lr and r["lr_last_str"] == args.lr_last
                       and r["seed"] in seeds), key=lambda r: r["seed"])
        print(f"Seeds  mpnn_ft  atom={args.atom}  lr={args.lr}  lr_last={args.lr_last}  "
              f"({len(rows)} finished runs: seeds {[r['seed'] for r in rows]})")
        if not rows:
            return
        for split in ("val", "test"):
            parts = []
            for m in METRICS:
                v = [r[f"{split}_{m}"] for r in rows]
                sd = st.stdev(v) if len(v) > 1 else 0.0
                parts.append(f"{m} {st.mean(v):.4f} +- {sd:.4f}")
            print(f"  {split:>4}: " + "   ".join(parts))
        print(f"  best epochs: {[r['best_epoch'] for r in rows]}")
        print(f"  ids(json) train/val/test: {sorted({r['sizes'] for r in rows})}")
        warn_mixed_splits(rows)


if __name__ == "__main__":
    main()
