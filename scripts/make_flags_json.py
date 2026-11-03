#!/usr/bin/env python
"""Build the reliability-flags file (mask_json) from the rBMRB-PANAV split jsons.

The training and inference scripts read the per-residue reliability flags ("{atom}_flag",
True = unreliable) from one json in the {field: {row: value}} layout. The PANAV split jsons
carry the same fields, so the flags file is their union:

    python scripts/make_flags_json.py --splits data/cs/train.json data/cs/val.json data/cs/test.json \
        --out data/cs/bmrb_cs_data_withoffset_panav_v2_flags_only.json

The flags of the PANAV jsons are used for every referencing scheme (LACS and BASELINE models are
evaluated on val/test with the same flags, as in the paper). After writing, the file is read back and
every flag list is compared with the split jsons; with --reference, also with an existing flags file.
"""
import argparse
import json
import sys

FIELDS = ["H_flag", "HA_flag", "N_flag", "CA_flag", "CB_flag", "C_flag"]


def by_entry(d):
    """{entry_ID: {field: flags}} of a {field: {row: value}} json."""
    return {d["entry_ID"][r]: {f: d[f][r] for f in FIELDS} for r in d["entry_ID"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", required=True, help="PANAV {train,val,test}.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference", default=None, help="an existing flags file to compare with (optional)")
    a = ap.parse_args()

    out = {f: {} for f in FIELDS}
    out["entry_ID"] = {}
    src = {}
    for path in a.splits:
        d = json.load(open(path))
        for e, flags in by_entry(d).items():
            if e in src and src[e] != flags:
                sys.exit(f"{e}: different flags in two split files")
            src[e] = flags
    for row, e in enumerate(sorted(src, key=lambda x: (len(x), x)), 1):
        out["entry_ID"][str(row)] = e
        for f in FIELDS:
            out[f][str(row)] = src[e][f]
    json.dump(out, open(a.out, "w"))

    back = by_entry(json.load(open(a.out)))                      # read back and check
    bad = [e for e in src if back.get(e) != src[e]]
    if bad or len(back) != len(src):
        sys.exit(f"read-back check FAILED: {len(bad)} entries differ, {len(back)} vs {len(src)} entries")
    n = len(src) * len(FIELDS)
    print(f"wrote {a.out}: {len(src)} entries x {len(FIELDS)} flag fields = {n} flag lists; read-back identical")

    if a.reference:
        ref = by_entry(json.load(open(a.reference)))
        missing = [e for e in src if e not in ref]
        diff = [(e, f) for e in src if e in ref for f in FIELDS if ref[e][f] != src[e][f]]
        print(f"reference {a.reference}: {len(ref)} entries; of ours: {len(missing)} missing there, "
              f"{len(diff)} of {n} flag lists differ")
        if missing or diff:
            sys.exit("reference check FAILED")


if __name__ == "__main__":
    main()
