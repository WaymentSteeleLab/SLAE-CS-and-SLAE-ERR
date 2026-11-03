#!/usr/bin/env python
"""SLAE-CS: predict per-residue backbone 15N chemical shifts for protein structures,
optionally with SLAE-ERROR estimates of the absolute error of each prediction.

    python predict.py --weights model/SLAE-CS-PANAV --pdb my_protein.pdb --out pred
    python predict.py --weights model/SLAE-CS-PANAV --error-model model/SLAE-ERROR --pdb structures/ --out pred

Input : one .pdb file or a directory of .pdb files. One protein per file; the first model of a
        multi-model file is used (split NMR ensembles first: scripts/inference_custom/split_nmr_models.py).
Output: <out>.csv   one row per residue: protein, index, chain, resnum, resname, N_mean, N_std,
                    [N_abs_error], and one column per seed (prolines are omitted: they have no amide N)
        <out>.json  {"seeds": [...], "pred": {protein: [n_seeds][L]}, "mask": ..., "mean": ..., "std": ...,
                     ["abs_error": ...]}

SLAE-ERROR (--error-model) needs the 10-seed SLAE-CS ensemble: its inputs are the residue embedding
of its own encoder, the per-residue std over the 10 seeds and the 10 seed predictions sorted and
centred on their mean (as in the training table, scripts/kfold/merge_oof_v3.py).

Needs the SLAE package (SLAE/ next to this file, or --slae_root) and its environment
(containers/slae/). Inference runs at batch size 1, as in the paper.
"""
import argparse
import csv
import glob
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # deterministic cuBLAS (before CUDA init)

HERE = Path(__file__).resolve().parent
PAD_VALUE = 1e-5                                    # padding value of the residue embeddings (as in training)
SHIFT_PRIOR = {"N": (100.0, 140.0, 119.0)}          # ppm window and output bias of the ShiftNet head
ERR_WINDOW = 40.0                                   # N shift window; SLAE-ERROR output clamp = [0, ERR_WINDOW / 2]
N_SEEDS_ERR = 10                                    # SLAE-ERROR was trained on 10-seed ensemble features


def residues_from_pdb(path):
    """(chain, resnum, resname) per CA residue of the first model."""
    out = []
    for line in open(path, errors="replace"):
        if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA":
            key = (line[21], line[22:27].strip(), line[17:20])
            if not out or out[-1] != key:
                out.append(key)
        if line.startswith("ENDMDL"):
            break
    return out


def weight_files(weights):
    w = Path(weights)
    if w.is_file():
        return [("model", w)]
    seeds = sorted(glob.glob(str(w / "seed*" / "model.safetensors")))
    if seeds:
        return [(Path(p).parent.name, Path(p)) for p in seeds]
    if (w / "model.safetensors").exists():
        return [("model", w / "model.safetensors")]
    sys.exit(f"no model.safetensors under {w}")


def ensemble_features(P, mask):
    """Inputs of SLAE-ERROR from the seed predictions, exactly as scripts/kfold/merge_oof_v3.py:
    P [n_seeds, L] float64, mask [L] bool -> std (ddof=1) [L] and sorted-centred [L, n_seeds], both
    float32, computed on the masked residues and 0.0 elsewhere."""
    import numpy as np
    mean = np.zeros(P.shape[1]); std = np.zeros(P.shape[1]); srt = np.zeros_like(P)
    if mask.any():
        mean[mask] = P[:, mask].mean(0)
        std[mask] = P[:, mask].std(0, ddof=1)
        srt[:, mask] = np.sort(P[:, mask], axis=0) - mean[mask]
    return std.astype(np.float32), srt.T.astype(np.float32)


def build_models(slae_root):
    """Return (SLAECS, SLAEError) module classes; imports the SLAE package from slae_root."""
    sys.path.insert(0, str(slae_root))
    import torch
    import torch.nn as nn
    from omegaconf import OmegaConf
    from SLAE.model.encoder import ProteinEncoder
    from SLAE.features.graph_featurizer import ProteinGraphFeaturizer
    from slae_cs.model.shiftnet import ShiftNet

    def encoder():
        cfg = OmegaConf.to_container(OmegaConf.load(Path(slae_root) / "SLAE/configs/encoder/protein_encoder.yaml"),
                                     resolve=True)
        cfg.pop("_target_", None)
        return ProteinEncoder(**cfg)

    def embed(self, batch):
        """featurise, encode, split the flat residue embeddings per protein, pad to [B, L, 128]"""
        batch = self.featuriser(batch)
        res_embedding = self.encoder(batch)["residue_embedding"]
        lengths = torch.bincount(batch.residue_batch, minlength=len(batch.id))
        seq_list, start = [], 0
        for length in lengths:
            seq_list.append(res_embedding[start:start + length.item()])
            start += length.item()
        x = nn.utils.rnn.pad_sequence(seq_list, batch_first=True, padding_value=PAD_VALUE)
        mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        return x, mask

    class SLAECS(nn.Module):
        """SLAE encoder + ShiftNet head; same submodule names (encoder.*, model.*) and forward as training."""

        def __init__(self):
            super().__init__()
            self.encoder = encoder()
            self.featuriser = ProteinGraphFeaturizer(radius=8.0, use_atom37=True)
            lo, hi, bias = SHIFT_PRIOR["N"]
            self.model = ShiftNet(emb_dim=128, width=256, ppm_window=(lo, hi), bias=bias, depth=4)

        def forward(self, batch):
            x, mask = embed(self, batch)
            return self.model(x, mask), mask

    class SLAEError(nn.Module):
        """SLAE-ERROR: its own SLAE encoder + ShiftNet error head (var_head) on
        [embedding (128) | ensemble std (1) | 10 sorted, centred seed predictions (10)];
        output = predicted |error| (ppm), clamped to [0, 20]. Same submodule names
        (encoder.*, var_head.*) and forward as error_head/chemical_shift_trainer_justerr_oof_sorted.py."""

        def __init__(self, n_sorted=N_SEEDS_ERR, hidden=256, depth=4):
            super().__init__()
            self.encoder = encoder()
            self.featuriser = ProteinGraphFeaturizer(radius=8.0, use_atom37=True)
            self.mu_err_clamp = 0.5 * ERR_WINDOW
            self.var_head = ShiftNet(emb_dim=128 + 1 + n_sorted, width=hidden, depth=depth,
                                     ppm_window=(-ERR_WINDOW / 2, ERR_WINDOW / 2), bias=0.0,
                                     analytic_centering=False)

        def forward(self, batch, std_feat=None, sorted_feat=None):
            """std_feat [B, L], sorted_feat [B, L, n_sorted] (zeros if None: used only to
            materialise the lazy layers)."""
            x, mask = embed(self, batch)
            if std_feat is None:
                std_feat = torch.zeros(mask.shape, device=x.device)
                sorted_feat = torch.zeros((*mask.shape, self.var_head.in_proj.in_features - 129), device=x.device)
            x = torch.cat([x, std_feat.unsqueeze(-1), sorted_feat], dim=-1)
            return self.var_head(x, mask).clamp(0.0, self.mu_err_clamp), mask

    return SLAECS, SLAEError


def load_batches(pdbs, tmp):
    """Featurisation input: one PDBDataModule inference batch per protein (batch size 1)."""
    from SLAE.datasets.datamodule import PDBDataModule
    (tmp / "structures").mkdir()
    ids = {}
    for p in pdbs:
        pid = p.name.split(".")[0]
        if pid in ids:
            sys.exit(f"{p} and {ids[pid]} map to the same protein id {pid}")
        ids[pid] = p
        shutil.copy(p, tmp / "structures" / f"{pid}.pdb")
    dm = PDBDataModule(pdb_dir=str(tmp / "structures"), processed_dir=str(tmp / "processed"),
                       batch_size=1, num_workers=0, overwrite=False, in_memory=False,
                       crop=False, rand_slice=False, inference_only=True)
    return ids, [b for b in dm.inference_dataloader() if b is not None]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--weights", required=True, help="dir with seedNN/model.safetensors, or one model.safetensors")
    ap.add_argument("--error-model", default=None, help="dir with the SLAE-ERROR model.safetensors (optional)")
    ap.add_argument("--pdb", required=True, help="a .pdb file or a directory of .pdb files")
    ap.add_argument("--out", required=True, help="output prefix (writes <out>.csv and <out>.json)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--slae_root", default=str(HERE), help="directory that contains the SLAE package")
    args = ap.parse_args()

    SLAECS, SLAEError = build_models(args.slae_root)
    import numpy as np
    import torch
    from safetensors.torch import load_file

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))
    seed_files = weight_files(args.weights)
    if args.error_model and len(seed_files) != N_SEEDS_ERR:
        sys.exit(f"--error-model needs the {N_SEEDS_ERR}-seed SLAE-CS ensemble, got {len(seed_files)} weight file(s)")

    src = Path(args.pdb)
    pdbs = [src] if src.is_file() else sorted(src.glob("*.pdb"))
    if not pdbs:
        sys.exit(f"no .pdb files in {src}")
    tmp = Path(tempfile.mkdtemp(prefix="slae_cs_"))
    try:
        ids, batches = load_batches(pdbs, tmp)

        torch.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        model = SLAECS().to(device).eval()            # eval(): dropout off
        with torch.no_grad():                         # materialise the encoder's lazy layers
            model(batches[0].to(device))

        seeds, preds, masks = [], {}, {}
        for name, f in seed_files:
            model.load_state_dict(load_file(str(f), device=str(device)), strict=True)
            seeds.append(name)
            with torch.inference_mode():
                for b in batches:
                    pred, mask = model(b.to(device))
                    pid = b.id[0]
                    preds.setdefault(pid, []).append(pred[0].cpu().double())
                    masks[pid] = mask[0].cpu()
            print(f"{name}: {len(batches)} proteins")

        abs_error = {}
        if args.error_model:
            err = SLAEError().to(device).eval()
            with torch.no_grad():
                err(batches[0].to(device))
            err.load_state_dict(load_file(str(Path(args.error_model) / "model.safetensors"), device=str(device)),
                                strict=True)
            with torch.inference_mode():
                for b in batches:
                    pid = b.id[0]
                    std, srt = ensemble_features(torch.stack(preds[pid]).numpy(), masks[pid].numpy())
                    mu, _ = err(b.to(device), torch.from_numpy(std)[None].to(device),
                                torch.from_numpy(srt)[None].to(device))
                    abs_error[pid] = mu[0].cpu().double()
            print(f"SLAE-ERROR: {len(batches)} proteins")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    out = {"seeds": seeds, "pred": {}, "mask": {}, "mean": {}, "std": {}}
    if abs_error:
        out["abs_error"] = {}
    rows = []
    missing = sorted(set(ids) - set(preds))
    for pid in sorted(preds):
        P = torch.stack(preds[pid])                                          # [n_seeds, L]
        seq_mask = masks[pid].clone()
        res = residues_from_pdb(ids[pid])
        if len(res) == P.shape[1]:
            seq_mask &= torch.tensor([r[2] != "PRO" for r in res])           # no amide N in proline
        else:
            res = [("", "", "")] * P.shape[1]
        mean = P.mean(0)
        std = P.std(0, unbiased=True) if len(seeds) > 1 else torch.zeros_like(mean)
        out["pred"][pid] = P.tolist()
        out["mask"][pid] = seq_mask.tolist()
        out["mean"][pid] = [float(m) if k else None for m, k in zip(mean, seq_mask)]
        out["std"][pid] = [float(s) if k else None for s, k in zip(std, seq_mask)]
        if abs_error:
            out["abs_error"][pid] = [float(e) if k else None for e, k in zip(abs_error[pid], seq_mask)]
        for i in range(P.shape[1]):
            if seq_mask[i]:
                extra = [f"{abs_error[pid][i]:.4f}"] if abs_error else []
                rows.append([pid, i, *res[i], f"{mean[i]:.4f}", f"{std[i]:.4f}", *extra,
                             *[f"{x:.4f}" for x in P[:, i]]])

    json.dump(out, open(args.out + ".json", "w"))
    with open(args.out + ".csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["protein", "index", "chain", "resnum", "resname", "N_mean", "N_std",
                    *(["N_abs_error"] if abs_error else []), *seeds])
        w.writerows(rows)
    print(f"{len(preds)} proteins, {len(rows)} residues, {len(seeds)} seeds"
          f"{' + SLAE-ERROR' if abs_error else ''} -> {args.out}.csv / {args.out}.json")
    if missing:
        print(f"WARNING no prediction (structure could not be processed): {missing}")


if __name__ == "__main__":
    main()
