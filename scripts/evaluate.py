#!/usr/bin/env python
"""Score predictions against the measured shifts (the protocol of the paper).

SLAE-CS (one or several per-seed prediction files of chemical_shift_predictor.py; several = the
mean over seeds, on the mask of the first file):
    python scripts/evaluate.py cs --gt data/cs/test.json --pred predictions_N_seed*_test.json

SLAE-ERROR (output of error_head/infer_oof_abs_sorted.py):
    python scripts/evaluate.py error --pred predictions_N_<tag>_err.json

Protocol (SLAE-CS): ground-truth values that are missing or non-finite, and residues without the
atom (N/H of proline, CB of glycine), are invalid; proteins without any valid value are skipped; a
residue counts if it is valid in the ground truth and in the prediction mask (reliability flags +
tag mask); RMSE / MAE over all such residues pooled. SLAE-ERROR: MAE |mu_err - |residual|| and
Spearman over the residues of oof_mask.
"""
import argparse
import json
import math


def parse_shift_list(lst, sequence, atom):
    vals, mask = [], []
    for i, x in enumerate(lst):
        bad = x is None
        if isinstance(x, str):
            xs = x.strip().lower()
            if xs in {"none", "null", "nan", "inf", "+inf", "-inf"}:
                bad = True
            else:
                try:
                    x = float(x)
                except ValueError:
                    bad = True
        if not bad and not math.isfinite(float(x)):
            bad = True
        if atom in ("N", "H") and sequence[i] == "P":
            bad = True
        if atom == "CB" and sequence[i] == "G":
            bad = True
        vals.append(0.0 if bad else float(x))
        mask.append(not bad)
    return vals, mask


def load_gt(path, atom):
    d = json.load(open(path))
    out = {}
    for r, lst in d[atom].items():
        if all(x is None or (isinstance(x, str) and x.lower() in ("none", "null")) for x in lst):
            continue
        out[d["entry_ID"][r].lower().removeprefix("bmr")] = parse_shift_list(lst, d["sequence"][r], atom)
    return out


def ranks(v):
    order = sorted(range(len(v)), key=v.__getitem__)
    r = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return cov / math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))


def score_cs(a):
    gt = load_gt(a.gt, a.atom)
    runs = [json.load(open(p)) for p in a.pred]
    se = ae = 0.0
    n = proteins = 0
    for pid in runs[0]["pred"]:
        key = pid.lower()
        if key not in gt:
            continue
        g, gm = gt[key]
        m0 = runs[0]["mask"][pid]
        used = 0
        for i in range(min(len(g), len(m0))):
            if gm[i] and m0[i]:
                p = sum(r["pred"][pid][i] for r in runs) / len(runs)
                se += (p - g[i]) ** 2
                ae += abs(p - g[i])
                n += 1
                used += 1
        proteins += used > 0
    print(f"{len(runs)} prediction file(s): proteins={proteins} residues={n} "
          f"RMSE={math.sqrt(se / n):.4f} MAE={ae / n:.4f}")


def score_error(a):
    d = json.load(open(a.pred[0]))
    x, y = [], []
    for pid in d["pred"]:
        for p, r, m in zip(d["pred"][pid], d["oof_resid"][pid], d["oof_mask"][pid]):
            if m:
                x.append(p)
                y.append(r)
    mae = sum(abs(p - r) for p, r in zip(x, y)) / len(x)
    print(f"residues={len(x)} MAE |mu_err - |resid||={mae:.4f} Spearman={spearman(x, y):.4f} "
          f"mean|resid|={sum(y) / len(y):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["cs", "error"])
    ap.add_argument("--pred", nargs="+", required=True)
    ap.add_argument("--gt", help="split json with the measured shifts (cs)")
    ap.add_argument("--atom", default="N")
    a = ap.parse_args()
    score_cs(a) if a.what == "cs" else score_error(a)


if __name__ == "__main__":
    main()
