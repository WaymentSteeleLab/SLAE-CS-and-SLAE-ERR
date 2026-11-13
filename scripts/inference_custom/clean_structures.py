#!/usr/bin/env python
"""Clean protein structures for predict.py with gemmi (https://gemmi.readthedocs.io): PDB or mmCIF in, PDB out.

    python scripts/inference_custom/clean_structures.py INPUT [INPUT ...] --out DIR

INPUT: structure files (.pdb, .ent, .cif, also gzipped) or directories of them. Each file is written to
DIR/<id>.pdb (<id>: file name up to the first "."), with only what SLAE-CS was trained on:
  - the first model;
  - polymer amino acids (waters, ligands, ions, DNA/RNA and hydrogens removed);
  - the first of alternative conformations;
  - one-character chain names (the PDB format), shortened by gemmi only if needed.
Needs gemmi (pip install gemmi).
"""
import argparse
from pathlib import Path

import gemmi

SUFFIXES = (".pdb", ".ent", ".cif", ".mmcif")


def inputs(paths):
    for p in map(Path, paths):
        files = sorted(p.iterdir()) if p.is_dir() else [p]
        yield from (f for f in files if f.is_file() and f.name.lower().removesuffix(".gz").endswith(SUFFIXES))


def clean(path):
    st = gemmi.read_structure(str(path))
    del st[1:]                                    # first model
    st.setup_entities()
    st.remove_hydrogens()
    st.remove_alternative_conformations()
    st.remove_ligands_and_waters()
    for chain in st[0]:                           # also drop non-amino-acid polymer residues (DNA, RNA)
        for i in reversed(range(len(chain))):
            info = gemmi.find_tabulated_residue(chain[i].name)
            if info is None or not info.is_amino_acid():
                del chain[i]
    st.remove_empty_chains()
    st.shorten_chain_names()
    return st


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    files = list(inputs(a.inputs))
    if not files:
        raise SystemExit("no structure files found")
    for f in files:
        st = clean(f)
        n = sum(len(ch) for ch in st[0])
        if n == 0:
            print(f"{f.name}: no amino acids left, skipped")
            continue
        out = a.out / (f.name.split(".")[0] + ".pdb")
        st.write_pdb(str(out), gemmi.PdbWriteOptions(minimal=True))
        print(f"{f.name} -> {out}: {n} residues in chains {' '.join(ch.name for ch in st[0])}")


if __name__ == "__main__":
    main()
