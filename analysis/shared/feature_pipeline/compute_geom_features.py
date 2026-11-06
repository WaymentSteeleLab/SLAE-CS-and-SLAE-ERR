#!/usr/bin/env python3
"""
Computes per-residue *physics-grounded geometric* descriptors that are known
determinants of backbone-amide 15N shift

For each residue i (nucleus = backbone amide N of residue i) we compute, from a
single static structure:
  - phi_i, psi_i               backbone dihedrals (continuous)          -> sin/cos
  - psi_prev                   psi of residue i-1 (dominant N determinant)-> sin/cos
  - phi_next                   phi of residue i+1                        -> sin/cos
  - chi1_i                     sidechain rotamer                        -> sin/cos (+flag)
  - prev_aa, next_aa           neighbour identities (categorical)
  - is_prepro                  next residue is Proline (0/1)
  - hb_HN_*                    H-bond descriptors (dist, cos angles, exists, DSSP-style
                                energy) for the amide-H (donor) of residue i — the one
                                H-bond that directly involves the backbone N-H being
                                predicted
  - rc_N                       ring-current shift contribution at N
  - pack_N8                    heavy-atom packing count within 8 A of N

We only care about the backbone amide N here, so instead of calling the SPARTA+/
UCBShift feature extractor's all-atom routines (`hbond_network` computes 3 H-bond
types per residue — alpha-H, amide-H, carbonyl-O — and `calc_ring_currents` computes
ring current at ~10 atom types per residue), this script defines its own thin
N-only wrappers below (`ring_current_N`, plus a direct call to the paper's
lower-level `NH_O_bond` primitive for just the amide donor bond). The underlying
physics is taken verbatim from `PDB_SPARTAp_DataReader` in
CSpred-ucbshift/spartap_features.py (Li et al. 2020, Chem. Sci. 11:3180): the same
Kabsch-Sander-style H-bond energy and >90 degree donor/acceptor angle criterion, and
the same Haigh-Mallion loop-current ring model with per-ring-type intensity factors
(Phe/Tyr/Trp x2/His) rather than a point-dipole at the ring centroid — just scoped
down to the N nucleus instead of computing every atom type.

Runs over every AF2 structure in the rBMRB/cluster_30p/structures_split_filtered/test
directory (no residue pre-filtering) and computes features for all standard
residues found in each structure.

Output: ../../../data/analysis/Fig4/geom_results/geom_features_af2.csv, keyed by (entry, resnum).
"""
import os, sys, glob, types, warnings
import numpy as np, pandas as pd
from Bio import PDB
from Bio.PDB import PDBParser
warnings.filterwarnings("ignore")

# ---- reuse the paper's own H-bond / ring-current code verbatim ----
# Bio.SubsMat was removed in modern biopython; spartap_features.py only needs it for
# blosum_nums(), which we never call, so stub it out rather than editing the paper's file.
if "Bio.SubsMat" not in sys.modules:
    _subsmat = types.ModuleType("Bio.SubsMat")
    _matinfo = types.ModuleType("Bio.SubsMat.MatrixInfo")
    _matinfo.blosum62 = {}
    sys.modules["Bio.SubsMat"] = _subsmat
    sys.modules["Bio.SubsMat.MatrixInfo"] = _matinfo

CSPRED_REPO = "../../../external/CSpred-ucbshift"
sys.path.insert(0, CSPRED_REPO)
from spartap_features import PDB_SPARTAp_DataReader

READER = PDB_SPARTAp_DataReader()

# ---- N-only ring current (Haigh-Mallion loop-current model) ----
# Ring atom definitions and per-ring intensity factors copied verbatim from
# PDB_SPARTAp_DataReader.calc_ring_currents; target_factors['N'] = 1.00 there.
RING_INTENSITY = {'PHE': 1.05, 'TYR': 0.92, 'TRP1': 1.04, 'TRP2': 0.90, 'HIS': 0.43}
N_TARGET_FACTOR = 1.00


def build_rings(residues):
    """Aromatic ring atom coordinates for every Phe/Tyr/Trp/His in the chain."""
    rings = []
    for res in residues:
        try:
            name = res.resname
            if name == 'PHE':
                rings.append(('PHE', np.array([res['CG'].coord, res['CD2'].coord,
                    res['CE2'].coord, res['CZ'].coord, res['CE1'].coord, res['CD1'].coord])))
            elif name == 'TYR':
                rings.append(('TYR', np.array([res['CG'].coord, res['CD2'].coord,
                    res['CE2'].coord, res['CZ'].coord, res['CE1'].coord, res['CD1'].coord])))
            elif name == 'TRP':
                rings.append(('TRP1', np.array([res['CD2'].coord, res['CE3'].coord,
                    res['CZ3'].coord, res['CH2'].coord, res['CZ2'].coord, res['CE2'].coord])))
                rings.append(('TRP2', np.array([res['CG'].coord, res['CD2'].coord,
                    res['CE2'].coord, res['NE1'].coord, res['CD1'].coord])))
            elif name == 'HIS':
                rings.append(('HIS', np.array([res['CG'].coord, res['ND1'].coord,
                    res['CE1'].coord, res['NE2'].coord, res['CD2'].coord])))
        except KeyError:
            pass  # incomplete side chain; skip this ring
    return rings


def ring_current_N(rings, N_coord):
    """Haigh-Mallion ring-current shift at N_coord: sum, bond-by-bond around each
    ring, of the signed projected area times the inverse-cube distance factor —
    same math as calc_ring_currents' inner loop, restricted to a single target atom."""
    shift = 0.0
    for label, ring_coords in rings:
        normal = np.cross(ring_coords[1] - ring_coords[0], ring_coords[-1] - ring_coords[0])
        normal = normal / np.linalg.norm(normal)
        o = N_coord + np.dot(normal, ring_coords[0] - N_coord) * normal
        n_atoms = len(ring_coords)
        G = 0.0
        for i in range(n_atoms):
            j = (i + 1) % n_atoms
            r_i, r_j = ring_coords[i] - o, ring_coords[j] - o
            d_i, d_j = ring_coords[i] - N_coord, ring_coords[j] - N_coord
            area = np.linalg.norm(np.cross(r_i, r_j)) / 2
            sign = np.sign(np.dot(np.cross(r_i, r_j), normal))
            d_ij = 1 / np.linalg.norm(d_i) ** 3 + 1 / np.linalg.norm(d_j) ** 3
            G += sign * area * d_ij
        shift += G * RING_INTENSITY[label] * N_TARGET_FACTOR
    return shift


def seq_neighbor_atoms(res_obj):
    """Atoms of the sequence-adjacent residues (i-1, i+1) in the same chain, used to
    exclude them from the H-bond acceptor search since they're covalently linked to
    residue i's backbone. Same residue-index lookup as hbond_network uses."""
    all_res = list(res_obj.get_parent().get_parent().get_residues())
    idx_i = all_res.index(res_obj)
    ip1_atoms, im1_atoms = [], []
    try:
        res_ip1 = all_res[idx_i + 1]
        if res_obj.get_full_id()[2] == res_ip1.get_full_id()[2]:
            ip1_atoms = list(res_ip1.get_atoms())
    except IndexError:
        pass
    try:
        res_im1 = all_res[idx_i - 1]
        if res_obj.get_full_id()[2] == res_im1.get_full_id()[2]:
            im1_atoms = list(res_im1.get_atoms())
    except IndexError:
        pass
    return im1_atoms, ip1_atoms

OUTDIR = "../../../data/analysis/Fig4/geom_results"   # read by merge_features.py; run from this folder

SOURCES = {
    "af2": dict(
        dir="../../../data/rBMRB/cluster_30p/structures_split_filtered/test",
        suffix=".pdb"),
}

AA3to1 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
          'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
          'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}

# first sidechain atom defining chi1 (N-CA-CB-X)
CHI1_X = {'ARG':'CG','ASN':'CG','ASP':'CG','CYS':'SG','GLN':'CG','GLU':'CG',
          'HIS':'CG','ILE':'CG1','LEU':'CG','LYS':'CG','MET':'CG','PHE':'CG',
          'PRO':'CG','SER':'OG','THR':'OG1','TRP':'CG','TYR':'CG','VAL':'CG1'}



def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    return np.degrees(np.arctan2(y, x))


def atomcoord(res, name):
    return res[name].coord if name in res else None


def process_structure(entry, pdb_path, want_resnums=None):
    """Return dict resnum -> feature dict.

    If want_resnums is None, compute features for every standard residue found
    in the structure's first chain (used when running over all 415 structures
    directly rather than a pre-filtered residue list).
    """
    model = PDBParser(QUIET=True).get_structure(entry, pdb_path)[0]
    # take the first chain that has standard residues
    chain = None
    for ch in model:
        if any(r.id[0] == ' ' for r in ch):
            chain = ch
            break
    residues = [r for r in chain if r.id[0] == ' ' and r.resname in AA3to1]
    by_num = {r.id[1]: r for r in residues}
    nums = sorted(by_num)
    if want_resnums is None:
        want_resnums = nums

    # aromatic rings for the whole chain, used by ring_current_N below
    rings = build_rings(residues)

    # H-bond search needs a neighbor-search tree over the whole chain so that the
    # paper's donor/acceptor search (nearest N/O within 5 A, excluding covalently
    # bonded neighbours, requiring >90 deg donor & acceptor angles) can look past the
    # immediate residue.
    nn_tree = PDB.NeighborSearch(list(chain.get_atoms()))

    # all heavy atoms for packing
    heavy = np.array([a.coord for a in chain.get_atoms()
                      if a.element != 'H']) if True else None

    out = {}
    for rn in want_resnums:
        if rn not in by_num:
            continue
        r = by_num[rn]
        f = {}
        N = atomcoord(r, 'N'); CA = atomcoord(r, 'CA'); C = atomcoord(r, 'C')
        rprev = by_num.get(rn - 1); rnext = by_num.get(rn + 1)

        # ---- backbone dihedrals ----
        def get_phi(res, prev):
            if prev is None or res is None:
                return np.nan
            Cp = atomcoord(prev, 'C'); n = atomcoord(res, 'N')
            ca = atomcoord(res, 'CA'); c = atomcoord(res, 'C')
            if any(x is None for x in (Cp, n, ca, c)):
                return np.nan
            return dihedral(Cp, n, ca, c)

        def get_psi(res, nxt):
            if nxt is None or res is None:
                return np.nan
            n = atomcoord(res, 'N'); ca = atomcoord(res, 'CA')
            c = atomcoord(res, 'C'); Nn = atomcoord(nxt, 'N')
            if any(x is None for x in (n, ca, c, Nn)):
                return np.nan
            return dihedral(n, ca, c, Nn)

        f['phi'] = get_phi(r, rprev)
        f['psi'] = get_psi(r, rnext)
        f['psi_prev'] = get_psi(rprev, r)
        f['phi_next'] = get_phi(rnext, r)

        # ---- chi1 ----
        chi1 = np.nan
        x = CHI1_X.get(r.resname)
        if x is not None:
            a = [atomcoord(r, nm) for nm in ('N', 'CA', 'CB', x)]
            if all(p is not None for p in a):
                chi1 = dihedral(*a)
        f['chi1'] = chi1

        # ---- neighbours ----
        f['prev_aa'] = AA3to1.get(rprev.resname, 'X') if rprev else 'NA'
        f['next_aa'] = AA3to1.get(rnext.resname, 'X') if rnext else 'NA'
        f['is_prepro'] = int(rnext is not None and rnext.resname == 'PRO')

        # ---- amide-H donor H-bond only (N-H...O), via the paper's own NH_O_bond ----
        # Returns [dist, cos(angle at acceptor), cos(angle at donor H), exists, energy];
        # exists requires both angles > 90 deg; energy is the Kabsch-Sander-style
        # electrostatic H-bond energy (same functional form DSSP uses).
        HNatom = None
        for at in ('H', '1H', 'H1', 'D', '1D', 'D1'):
            if at in r:
                HNatom = r[at]
                break
        if HNatom is not None and 'N' in r:
            im1_atoms, ip1_atoms = seq_neighbor_atoms(r)
            try:
                hn = READER.NH_O_bond(nn_tree, r, im1_atoms, ip1_atoms, 5.0,
                                       r['N'], HNatom, ['N', 'O'], efilt=False)
            except Exception:
                hn = [0] * 5
        else:
            hn = [0] * 5
        d, cos_a, cos_h, exists, energy = hn
        f['hb_HN_dist'] = float(d) if exists else np.nan
        f['hb_HN_cosA'] = float(cos_a) if exists else np.nan
        f['hb_HN_cosH'] = float(cos_h) if exists else np.nan
        f['hb_HN_exists'] = int(exists)
        f['hb_HN_energy'] = float(energy) if exists else np.nan

        # ---- ring current at N only (Haigh-Mallion loop-current model) ----
        f['rc_N'] = ring_current_N(rings, N) if N is not None else np.nan

        # ---- local packing ----
        if N is not None and heavy is not None and len(heavy):
            f['pack_N8'] = int((np.linalg.norm(heavy - N, axis=1) < 8.0).sum())
        else:
            f['pack_N8'] = np.nan

        out[rn] = f
    return out


def main():
    for src, cfg in SOURCES.items():
        pdb_files = sorted(glob.glob(os.path.join(cfg['dir'], '*' + cfg['suffix'])))
        rows = []
        n_ok = n_fail = 0
        for path in pdb_files:
            entry = os.path.basename(path)[:-len(cfg['suffix'])]
            try:
                feats = process_structure(entry, path)
            except Exception as ex:
                n_fail += 1
                print(f"  ! {src} {entry}: {ex}", file=sys.stderr)
                continue
            n_ok += 1
            for rn, f in feats.items():
                rows.append(dict(entry=entry, resnum=rn, **f))
        out = pd.DataFrame(rows)
        # encode angles as sin/cos
        for ang in ['phi', 'psi', 'psi_prev', 'phi_next', 'chi1']:
            rad = np.deg2rad(out[ang])
            out[f'{ang}_sin'] = np.sin(rad)
            out[f'{ang}_cos'] = np.cos(rad)
            out[f'{ang}_present'] = out[ang].notna().astype(int)
        outpath = os.path.join(OUTDIR, f"geom_features_{src}.csv")
        out.to_csv(outpath, index=False)
        print(f"[{src}] proteins_ok={n_ok}  proteins_failed={n_fail}  "
              f"rows={len(out)}  -> {outpath}")


if __name__ == "__main__":
    main()
