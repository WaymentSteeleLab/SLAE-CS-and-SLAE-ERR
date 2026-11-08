"""Extract per-atom MACE-OFF descriptors for the AF2 (ColabFold) structures.

Needs the MACE-OFF23 large weights (Academic Software License, not redistributed); run
baselines/setup_third_party.py --mace --accept-asl, or build baselines/mace/mace.def (which sets $MACE_MODEL).

Follows Bojan et al., "Representing local protein environments with machine
learning force fields" (arXiv:2505.23354; github.com/mb012/MLFF_representation):
MACE-OFF23 large, invariant features concatenated over all layers, computed on a
local environment made of every residue with any atom within `rmax` of the
focus residue.

Output: one .npz per structure with keys
    resnum, resname            (L,)
    <atom> for atom in ATOMS   (L, D) float32, NaN where the atom is absent
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from Bio.PDB import PDBParser
from scipy.spatial import cKDTree
from mace.calculators import mace_off

ATOMS = ["N"]


def load_structure(path):
    """Return per-atom arrays for the first chain (standard residues only)."""
    structure = PDBParser(QUIET=True).get_structure("x", str(path))
    chain = next(structure[0].get_chains())
    res_idx, names, elements, coords = [], [], [], []
    resnums, resnames = [], []
    for i, res in enumerate(r for r in chain if r.id[0] == " "):
        resnums.append(res.id[1])
        resnames.append(res.get_resname())
        for atom in res:
            res_idx.append(i)
            names.append(atom.get_id())
            elements.append(atom.element.capitalize())
            coords.append(atom.coord)
    return (np.array(res_idx), np.array(names), np.array(elements),
            np.array(coords, dtype=np.float64), np.array(resnums), np.array(resnames))


def descriptors_crop(calc, res_idx, elements, coords, rmax):
    """One MACE call per residue on its cropped environment (paper protocol)."""
    tree = cKDTree(coords)
    n_res = res_idx.max() + 1
    out = [None] * len(res_idx)
    for r in range(n_res):
        focus = np.where(res_idx == r)[0]
        hits = tree.query_ball_point(coords[focus], rmax)
        env_res = np.unique(res_idx[np.concatenate(hits).astype(int)])
        env = np.where(np.isin(res_idx, env_res))[0]
        atoms = Atoms(symbols=elements[env].tolist(), positions=coords[env])
        desc = calc.get_descriptors(atoms, invariants_only=True, num_layers=-1)
        pos = {a: k for k, a in enumerate(env)}
        for a in focus:
            out[a] = desc[pos[a]]
    return np.stack(out)


def descriptors_whole(calc, elements, coords):
    """Single MACE call on the full protein (faster; not the paper protocol)."""
    atoms = Atoms(symbols=elements.tolist(), positions=coords)
    return calc.get_descriptors(atoms, invariants_only=True, num_layers=-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pdb_dir", required=True)
    p.add_argument("--list", required=True, help="text file, one PDB filename per line")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--model", default="large", help="'large' or path to a .model file")
    p.add_argument("--mode", choices=["crop", "whole"], default="crop")
    p.add_argument("--rmax", type=float, default=5.0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    calc = mace_off(model=args.model, device=args.device, default_dtype="float64")

    names = [l.strip() for l in open(args.list) if l.strip()]
    for k, fname in enumerate(names):
        out_path = out_dir / (Path(fname).stem + ".npz")
        if out_path.exists():
            continue
        t0 = time.time()
        try:
            res_idx, atom_names, elements, coords, resnums, resnames = load_structure(Path(args.pdb_dir) / fname)
            if args.mode == "crop":
                desc = descriptors_crop(calc, res_idx, elements, coords, args.rmax)
            else:
                desc = descriptors_whole(calc, elements, coords)
            n_res, dim = len(resnums), desc.shape[1]
            arrays = {"resnum": resnums, "resname": resnames}
            for a in ATOMS:
                arr = np.full((n_res, dim), np.nan, dtype=np.float32)
                sel = np.where(atom_names == a)[0]
                arr[res_idx[sel]] = desc[sel].astype(np.float32)
                arrays[a] = arr
            np.savez_compressed(out_path, **arrays)
            print(f"[{k+1}/{len(names)}] {fname}: L={n_res} D={dim} {time.time()-t0:.1f}s", flush=True)
        except Exception as e:
            print(f"[{k+1}/{len(names)}] {fname}: FAILED {e!r}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
