#!/usr/bin/env python
"""Smoke test: run predict.py on the example structures and compare with the reference output.

    python tests/smoke_test.py [--quick] [--device auto|cuda|cpu] [--tol 1e-3]
                               [--weights model/SLAE-CS-PANAV] [--error-model model/SLAE-ERR]

Run it in the SLAE-CS environment (Apptainer image or conda, see README), from the repository root,
after `python scripts/download_from_hf.py --no-data` (weights only).
examples/structures/ holds 3 small proteins of the PANAV test set (AlphaFold2 models from
hws-lab/rBMRB); examples/expected/example_gpu.csv is the predict.py output for them in the
environment of the paper (slae.sif, NVIDIA A100). The test checks that the output has the same
residues and columns, and that every value (N_mean, N_std, N_abs_error and the 10 seeds; written with
4 decimals) agrees within --tol ppm. Other GPUs, and the CPU, change the values by floating-point
rounding only. --quick runs 34164_1 alone.
"""
import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
KEY = ("protein", "index", "chain", "resnum", "resname")
QUICK = "34164_1"


def read(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    return {tuple(r[k] for k in KEY): r for r in rows}, (list(rows[0]) if rows else [])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help=f"run {QUICK} only")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--tol", type=float, default=1e-3, help="largest allowed difference (ppm)")
    ap.add_argument("--weights", default=str(ROOT / "model" / "SLAE-CS-PANAV"))
    ap.add_argument("--error-model", default=str(ROOT / "model" / "SLAE-ERR"))
    a = ap.parse_args()
    for w in (a.weights, a.error_model):
        if not Path(w).exists():
            sys.exit(f"{w} not found: download the weights first (python scripts/download_from_hf.py --no-data)")

    pdb = EXAMPLES / "structures" / (f"{QUICK}.pdb" if a.quick else "")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "example"
        cmd = [sys.executable, str(ROOT / "predict.py"), "--weights", a.weights, "--error-model", a.error_model,
               "--pdb", str(pdb), "--out", str(out), "--device", a.device]
        print("running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        got, got_cols = read(f"{out}.csv")

    ref, ref_cols = read(EXAMPLES / "expected" / "example_gpu.csv")
    if a.quick:
        ref = {k: v for k, v in ref.items() if k[0] == QUICK}
    problems = []
    if got_cols != ref_cols:
        problems.append(f"columns differ: {got_cols} vs {ref_cols}")
    if set(got) != set(ref):
        problems.append(f"residues differ: {len(set(got) - set(ref))} extra, {len(set(ref) - set(got))} missing")
    values = [c for c in ref_cols if c not in KEY]
    worst = {c: max((abs(float(got[k][c]) - float(r[c])) for k, r in ref.items() if k in got), default=0.0)
             for c in values}
    print(f"{len({k[0] for k in got})} proteins, {len(got)} residues; largest difference from the reference:")
    print("  " + ", ".join(f"{c} {worst[c]:.1e}" for c in ("N_mean", "N_std", "N_abs_error")) +
          f", seeds {max(worst[c] for c in values if c.startswith('seed')):.1e} ppm")
    bad = [c for c in values if worst[c] > a.tol]
    if bad:
        problems.append(f"differences above {a.tol} ppm in {bad}")
    if problems:
        sys.exit("FAILED: " + "; ".join(problems))
    print(f"passed (tolerance {a.tol} ppm)")


if __name__ == "__main__":
    main()
