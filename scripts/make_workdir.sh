#!/usr/bin/env bash
# Make a working directory for one run. The scripts read SLAE/configs, data/cs/, structures/ and the
# {train,val,test}.list files relative to the directory they are started from, and write their
# checkpoints / predictions there, so every run gets its own directory with links to the shared files.
#
#   scripts/make_workdir.sh <dir> [--data-root DIR] [--structures LIST]
#
#   --data-root   where download_from_hf.py put the data (default: the repository root)
#   --structures  link only the structures of the ids in LIST (e.g. splits/panav/test.list for
#                 test-set inference, splits/kfold/heldout_fold0.list for out-of-fold inference);
#                 default: all of <data-root>/structures
set -euo pipefail

[ $# -ge 1 ] || { sed -n 2,11p "$0"; exit 1; }
WD="$1"; shift
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="$REPO_ROOT"
LIST=""
while [ $# -gt 0 ]; do
    case "$1" in
        --data-root) DATA_ROOT="$(cd "$2" && pwd)"; shift 2 ;;
        --structures) LIST="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"; shift 2 ;;
        *) echo "unknown option $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$WD"
WD="$(cd "$WD" && pwd)"
ln -sfn "$REPO_ROOT/SLAE" "$WD/SLAE"
ln -sfn "$DATA_ROOT/data" "$WD/data"
[ -f "$DATA_ROOT/autoencoder.ckpt" ] && ln -sfn "$DATA_ROOT/autoencoder.ckpt" "$WD/autoencoder.ckpt"
for s in train val test; do [ -f "$DATA_ROOT/$s.list" ] && ln -sfn "$DATA_ROOT/$s.list" "$WD/$s.list"; done
if [ -z "$LIST" ]; then
    ln -sfn "$DATA_ROOT/structures" "$WD/structures"
else
    mkdir -p "$WD/structures"
    n=0
    while read -r id; do
        [ -z "$id" ] && continue
        [ -e "$DATA_ROOT/structures/$id.pdb" ] || { echo "missing structure $id" >&2; exit 1; }
        ln -sfn "$(readlink -f "$DATA_ROOT/structures/$id.pdb")" "$WD/structures/$id.pdb"; n=$((n + 1))
    done < "$LIST"
    echo "linked $n structures"
fi
echo "working directory: $WD"
