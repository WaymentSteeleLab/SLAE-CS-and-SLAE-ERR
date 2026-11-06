# Per-frame DSSP (8-state) + per-frame phi/psi for every gated residue of the 48 MD proteins.
# Gate = GT 15N not None + prediction mask + not Pro.
# Usage: python compute_md_dssp8_phipsi_gated_allframes.py [output.npz]. Takes about 30 min with 6 processes.
# Protein (column) order depends on which worker finishes first; downstream code looks residues up by (entry, resnum).
import glob
import json
import os
import re
import sys
from multiprocessing import Pool

import numpy as np
import mdtraj as mdt

FIG4 = "../../data/analysis/Fig4"
RES = "../../data/md_1us"
PRED = f"{FIG4}/results/predictions_N_seed_ensemble_10seed.json"
GT = "../../data/rBMRB/cluster_30p/PANAV/test.json"
OUT = sys.argv[1] if len(sys.argv) > 1 else f"{FIG4}/per_protein_analysis/abnormal_coil/md_dssp8_phipsi_gated_allframes_panav10seedens.npz"
_FRAME_RE = re.compile(r"^frame_?(\d+)")


def _frames(e):
    return sorted((p for p in glob.glob(f"{RES}/{e}/frame*.pdb") if _FRAME_RE.match(os.path.basename(p))),
                  key=lambda p: int(_FRAME_RE.match(os.path.basename(p)).group(1)))


def load_traj(paths):
    frames = [mdt.load(p) for p in paths]
    return mdt.Trajectory(xyz=np.concatenate([f.xyz for f in frames]), topology=frames[0].topology)


def gated_resnums():
    pred, gt = json.load(open(PRED)), json.load(open(GT))
    entries = sorted(d for d in os.listdir(RES) if os.path.isdir(f"{RES}/{d}"))   # the 48 MD proteins
    key_of = {v: k for k, v in gt["entry_ID"].items()}
    out = {}
    for e in entries:
        k = key_of.get(e)
        if k is None or e not in pred["mean"]:
            continue
        seq, shifts = gt["sequence"][k], gt["N"][k]
        n = len(seq)
        pm = np.array(pred["mask"][e], dtype=bool)
        if len(pred["mean"][e]) < n or len(pm) < n:
            continue
        m = np.array([r != "P" for r in seq]) & np.array([s is not None for s in shifts]) & pm[:n]
        out[e] = (np.where(m)[0] + 1).tolist()
    return out


def work(args):
    e, want = args
    traj = load_traj(_frames(e))                       # all frames incl. equilibration
    top = traj.top
    dssp = mdt.compute_dssp(traj, simplified=False)     # [n_frames, n_res]
    phi_idx, phi = mdt.compute_phi(traj)
    psi_idx, psi = mdt.compute_psi(traj)
    phi_col = {top.atom(r[2]).residue.index + 1: k for k, r in enumerate(phi_idx)}
    psi_col = {top.atom(r[1]).residue.index + 1: k for k, r in enumerate(psi_idx)}
    keep = [rn for rn in want if rn in phi_col and rn in psi_col and rn <= top.n_residues]
    return dict(entry=e, resnum=np.array(keep), resname=[top.residue(rn - 1).name for rn in keep],
                dssp=np.stack([dssp[:, rn - 1] for rn in keep], 1).astype("U1"),
                phi=np.degrees(np.stack([phi[:, phi_col[rn]] for rn in keep], 1)).astype(np.float32),
                psi=np.degrees(np.stack([psi[:, psi_col[rn]] for rn in keep], 1)).astype(np.float32))


if __name__ == "__main__":
    g = gated_resnums()
    print(len(g), "proteins,", sum(map(len, g.values())), "gated residues", flush=True)
    res = []
    with Pool(6) as pool:
        for r in pool.imap_unordered(work, list(g.items())):
            print(r["entry"], r["dssp"].shape, flush=True)
            res.append(r)
    nf = {r["dssp"].shape[0] for r in res}
    assert len(nf) == 1, nf
    ent = np.concatenate([[r["entry"]] * len(r["resnum"]) for r in res])
    np.savez_compressed(OUT, entry=ent.astype(str), resnum=np.concatenate([r["resnum"] for r in res]),
                        resname=np.concatenate([r["resname"] for r in res]).astype(str),
                        dssp=np.concatenate([r["dssp"] for r in res], 1),
                        phi=np.concatenate([r["phi"] for r in res], 1),
                        psi=np.concatenate([r["psi"] for r in res], 1))
    print("wrote", OUT)
