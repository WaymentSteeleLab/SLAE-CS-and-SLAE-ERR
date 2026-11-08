#!/usr/bin/env python
"""Download the third-party files of the baselines into external/, at the commits used in the paper.

None of these files are stored in this repository or in our model repositories.

    python baselines/setup_third_party.py --proteinmpnn            # MIT
    python baselines/setup_third_party.py --mace --accept-asl      # Academic Software License

The ESM3 weights are not handled here: baselines/esm3/extract_esm3.py downloads them from Hugging Face, under
the terms stated there (see baselines/README.md).
"""
import argparse
import hashlib
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MPNN = "https://raw.githubusercontent.com/dauparas/ProteinMPNN/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57"
MACE = "https://raw.githubusercontent.com/ACEsuit/mace-off/91a78c5a9c300d1104700d9352c8bfe449227737"
FILES = {
    "proteinmpnn": [
        (f"{MPNN}/protein_mpnn_utils.py", "ProteinMPNN/protein_mpnn_utils.py",
         "74c8f9b7553422a7a0bbd705874844ee103c8926c2c96f154a87e0b824071e1b"),
        (f"{MPNN}/vanilla_model_weights/v_48_020.pt", "ProteinMPNN/v_48_020.pt",
         "c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd"),
        (f"{MPNN}/LICENSE", "ProteinMPNN/LICENSE", None),
    ],
    "mace": [
        (f"{MACE}/mace_off23/MACE-OFF23_large.model", "mace-off/MACE-OFF23_large.model",
         "a29e397dbf3e7a24ac50a9b0dfc919bd5a62efa346f5895a6237b0950c1d76f4"),
        (f"{MACE}/LICENSE.md", "mace-off/LICENSE.md", None),
    ],
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url, rel, want, dest):
    dst = dest / rel
    if dst.exists() and (want is None or sha256(dst) == want):
        print(f"{dst}: present")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    urllib.request.urlretrieve(url, tmp)
    got = sha256(tmp)
    if want is not None and got != want:
        tmp.unlink()
        raise SystemExit(f"{url}: sha256 {got}, expected {want}")
    tmp.rename(dst)
    print(f"{dst}: downloaded")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proteinmpnn", action="store_true")
    ap.add_argument("--mace", action="store_true")
    ap.add_argument("--accept-asl", action="store_true",
                    help="you accept the Academic Software License of MACE-OFF23 (academic, non-commercial use only)")
    ap.add_argument("--dest", type=Path, default=REPO_ROOT / "external")
    a = ap.parse_args()
    if not (a.proteinmpnn or a.mace):
        ap.error("nothing to do: give --proteinmpnn and / or --mace")
    if a.mace and not a.accept_asl:
        ap.error("MACE-OFF23 is under the Academic Software License "
                 "(https://github.com/ACEsuit/mace-off/blob/main/LICENSE.md): read it, then add --accept-asl")
    for name, chosen in (("proteinmpnn", a.proteinmpnn), ("mace", a.mace)):
        if chosen:
            for url, rel, want in FILES[name]:
                fetch(url, rel, want, a.dest)


if __name__ == "__main__":
    main()
