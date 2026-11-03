#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build the K-fold OOF splits of the PANAV train.list. Clusters of result_cluster.tsv (structure +
sequence similarity) are never split across folds; otherwise a homologous
structure could leak from one fold's training part into another fold's held-out part and make its
OOF residuals look too good.

result_cluster.tsv: two tab-separated columns (cluster_representative_id, member_id), one row per
id of train.list; the assertions below check that it covers train.list exactly.

Assignment: greedy first-fit-decreasing bin packing -- clusters sorted by
size descending (ties broken by a fixed-seed shuffle), each cluster assigned
whole to whichever of the K fold buckets currently holds the fewest ids. This
keeps fold sizes close to balanced (~N/K each) without ever splitting a
cluster.

Outputs (into --out_dir):
  fold_assignment.tsv          -- id, fold  (for bookkeeping/audit)
  train_fold{k}.list            -- ids from all folds EXCEPT k (K-1 folds)
  heldout_fold{k}.list          -- ids from fold k only (the OOF target set)
  fold_list.txt                 -- "0\n1\n...\n{K-1}\n", one fold per job

Run once (CPU only):
    python scripts/kfold/make_kfold_splits.py --train_list train.list \
        --clusters data/structures/train_cluster/result_cluster.tsv --out_dir splits/kfold
"""
import argparse
import random
from collections import defaultdict
from pathlib import Path

K = 5
SEED = 42


def main():
    ap = argparse.ArgumentParser(description="cluster-respecting K-fold splits of train.list")
    ap.add_argument("--train_list", required=True, help="train.list (4647 PANAV training ids)")
    ap.add_argument("--clusters", required=True, help="result_cluster.tsv (representative, member)")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    HERE = Path(a.out_dir)
    HERE.mkdir(parents=True, exist_ok=True)
    CLUSTER_TSV, TRAIN_LIST = Path(a.clusters), Path(a.train_list)
    train_ids = [l.strip() for l in TRAIN_LIST.read_text().splitlines() if l.strip()]
    train_id_set = set(train_ids)

    clusters = defaultdict(list)  # rep_id -> [member_id, ...]
    seen_members = set()
    with open(CLUSTER_TSV) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rep, member = line.split("\t")
            clusters[rep].append(member)
            seen_members.add(member)

    # sanity checks -- must exactly cover train.list, nothing more/less
    missing = train_id_set - seen_members
    extra = seen_members - train_id_set
    assert not missing, f"{len(missing)} train ids missing from result_cluster.tsv, e.g. {list(missing)[:5]}"
    assert not extra, f"{len(extra)} ids in result_cluster.tsv not in train.list, e.g. {list(extra)[:5]}"

    cluster_list = list(clusters.items())  # [(rep_id, [members...]), ...]
    rng = random.Random(SEED)
    rng.shuffle(cluster_list)                      # break size-ties reproducibly
    cluster_list.sort(key=lambda kv: len(kv[1]), reverse=True)  # stable -> shuffle order kept within ties

    fold_ids = [[] for _ in range(K)]
    fold_counts = [0] * K
    for rep, members in cluster_list:
        target = min(range(K), key=lambda k: fold_counts[k])
        fold_ids[target].extend(members)
        fold_counts[target] += len(members)

    print(f"{len(cluster_list)} clusters over {len(train_ids)} train ids -> {K} folds")
    for k, c in enumerate(fold_counts):
        print(f"  fold {k}: {c} ids")

    # fold_assignment.tsv (audit trail)
    with open(HERE / "fold_assignment.tsv", "w") as f:
        for k, ids in enumerate(fold_ids):
            for i in ids:
                f.write(f"{i}\t{k}\n")

    for k in range(K):
        heldout = set(fold_ids[k])
        train_other = [i for i in train_ids if i not in heldout]
        assert len(train_other) + len(heldout) == len(train_ids)
        (HERE / f"heldout_fold{k}.list").write_text("\n".join(sorted(heldout)) + "\n")
        (HERE / f"train_fold{k}.list").write_text("\n".join(train_other) + "\n")

    (HERE / "fold_list.txt").write_text("\n".join(str(k) for k in range(K)) + "\n")
    print(f"\nWrote train_fold{{0..{K-1}}}.list, heldout_fold{{0..{K-1}}}.list, "
          f"fold_assignment.tsv, fold_list.txt to {HERE}")


if __name__ == "__main__":
    main()
