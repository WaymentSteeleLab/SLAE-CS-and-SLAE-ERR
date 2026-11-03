#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SLAE-CS inference: predict per-residue shifts for every structure in ./structures with one trained
chemical_shift_trainer.py checkpoint. Writes predictions_{atom}_{method}.json:
{"pred": {pdb: [...]}, "mask": {pdb: [...]}}. The mask is the residue mask, ANDed with the optional
reliability flags (mask_json) and tag mask (tag_mask_csv). Always runs at batch size 1.

Run from the repository root (hydra config configs/slae_cs_infer.yaml; paths absolute):
    PYTHONPATH=$PWD python slae_cs/inference/chemical_shift_predictor.py ckpt=<model.safetensors or .ckpt> atom=N method=<tag> \
        [mask_json=<flags.json>] [tag_mask_csv=<tag_labels.csv>]
"""

import os, json, warnings
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import hydra

from SLAE.model.encoder import ProteinEncoder
from slae_cs.model.shiftnet import ShiftNet
from SLAE.datasets.datamodule import PDBDataModule
from SLAE.features.graph_featurizer import ProteinGraphFeaturizer

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# needed (with torch.use_deterministic_algorithms in cli()) for deterministic cuBLAS matmuls;
# must be set before the CUDA context is created.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# ------------------------------------------------------------------ #
#  Paths & constants                                                 #
# ------------------------------------------------------------------ #
cwd = os.getcwd()
ENCODER_CFG = Path(f"{cwd}/SLAE/configs/encoder/protein_encoder.yaml")
PAD_VALUE   = 1e-5


def _to_plain_dict(cfg_node) -> dict:
    """Flatten an OmegaConf node to a plain Python dict (recursive)."""
    return OmegaConf.to_container(cfg_node, resolve=True, structured_config_mode=False)


# Reliability flags ("{atom}_flag" = True where unreliable), inverted to True = keep.
# Same as chemical_shift_trainer._load_extra_mask_dict.
def _load_extra_mask_dict(mask_json_path: Path, atom: str = "N") -> Dict[str, torch.Tensor]:
    with open(mask_json_path) as f:
        rbmrb = json.load(f)

    mask_key = f"{atom}_flag"
    out = {}
    for ref_id, mask_lst in rbmrb[mask_key].items():
        pdb_id = rbmrb["entry_ID"][ref_id].lower().removeprefix("bmr")
        out[pdb_id] = torch.tensor([not bool(m) for m in mask_lst], dtype=torch.bool)
    return out


# Tag/linker keep mask from tag_labels.csv: 'A'/'P' = native residue (keep), 'T'/'t' = tag /
# linker residue. Same as chemical_shift_trainer._load_tag_mask_dict.
def _load_tag_mask_dict(csv_path: Path, id_col: str = "BMRB_entry",
                        label_col: str = "label") -> Dict[str, torch.Tensor]:
    import pandas as pd
    df = pd.read_csv(csv_path)
    out = {}
    for _, row in df.iterrows():
        label = row.get(label_col)
        if not isinstance(label, str) or not label:
            continue
        pdb_id = str(row[id_col]).lower().removeprefix("bmr")
        out[pdb_id] = torch.tensor([c in ("A", "P") for c in label], dtype=torch.bool)
    return out


# AND several pdb_id -> bool-mask dicts; a pdb_id missing from one source is not gated by it.
def _and_mask_dicts(*dicts: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    dicts = [d for d in dicts if d]
    if not dicts:
        return {}
    ids = set().union(*(d.keys() for d in dicts))
    out = {}
    for pdb_id in ids:
        m = None
        for d in dicts:
            if pdb_id not in d:
                continue
            if m is None:
                m = d[pdb_id].clone()
            else:
                n = min(len(m), len(d[pdb_id]))
                m = m[:n] & d[pdb_id][:n]
        out[pdb_id] = m
    return out


def build_encoder_from_files(cfg_path: Path) -> ProteinEncoder:
    """Build the ProteinEncoder architecture only; cli() materialises its lazy layers and then
    loads the trained weights."""
    full_cfg = OmegaConf.load(cfg_path)
    enc_cfg  = _to_plain_dict(full_cfg)
    enc_cfg.pop("_target_", None)
    return ProteinEncoder(**enc_cfg).to("cuda")


# ------------------------------------------------------------------ #
#  LightningModule (inference-only)                                  #
# ------------------------------------------------------------------ #
class CSLightning(pl.LightningModule):
    """Encoder + ShiftNet head with the training module's parameter names; forward() only."""

    def __init__(self, embedding: str = "residue_embedding", atom: str = "N"):
        super().__init__()

        self.encoder = build_encoder_from_files(ENCODER_CFG)
        self.embedding = embedding
        self.featuriser = ProteinGraphFeaturizer(radius=8.0, use_atom37=True)
        self.atom = atom

        if atom == "N":
            min_shift, max_shift, bias = 100.0, 140.0, 119.0
        elif atom == "C":
            min_shift, max_shift, bias = 163.0, 186.0, 175.0
        elif atom == "H":
            min_shift, max_shift, bias = 4, 12, 8
        elif atom == "CA":
            min_shift, max_shift, bias = 40, 72, 56
        elif atom == "CB":
            min_shift, max_shift, bias = 14, 80, 38
        elif atom == "HA":
            min_shift, max_shift, bias = 2, 7, 4.5
        print(f"ShiftNet: atom={atom}  ppm_window=({min_shift}, {max_shift})  bias={bias}")

        # analytic_centering only affects ShiftNet.loss(), not forward()
        self.model = ShiftNet(emb_dim=128, width=256,
                              ppm_window=(min_shift, max_shift),
                              bias=bias, depth=4)

    def forward(self, batch):
        batch = self.featuriser(batch)

        if self.embedding == "residue_embedding":
            res_embedding = self.encoder(batch)[self.embedding]
        else:  # node_embedding -- use N atoms only
            node_embedding = self.encoder(batch)[self.embedding]   # [N_atom, 128]
            n_mask         = batch.atom37_type.eq(0)
            res_embedding  = node_embedding[n_mask]
            assert res_embedding.size(0) == batch.residue_batch.size(0)

        residue_batch = batch.residue_batch
        batch_size = len(batch.id)
        lengths = torch.bincount(residue_batch, minlength=batch_size)  # [B]
        seq_list, start = [], 0
        for length in lengths:
            seq_len = length.item()
            seq_list.append(res_embedding[start:start + seq_len])
            start += seq_len

        res_emb_padded = nn.utils.rnn.pad_sequence(seq_list, batch_first=True,
                                                   padding_value=PAD_VALUE)
        max_len = res_emb_padded.size(1)
        mask = torch.arange(max_len, device=res_emb_padded.device).unsqueeze(0) < lengths.unsqueeze(1)
        pred = self.model(res_emb_padded, mask)
        return pred, mask


# ------------------------------------------------------------------ #
#  CLI                                                               #
# ------------------------------------------------------------------ #
@hydra.main(version_base=None, config_path="../../configs", config_name="slae_cs_infer")
def cli(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    ckpt_path  = Path(cfg.ckpt)
    # Always 1: with padding, ShiftNet's depth-wise conv (kernel 5, before any masking) mixes pad
    # vectors into the last 2 residues of the shorter protein (~0.15 ppm), so predictions would
    # depend on batch composition.
    batch_size = 1
    if cfg.batch != 1:
        print(f"[INFER] ignoring cfg.batch={cfg.batch} -- inference always runs with batch size 1")
    embed      = cfg.embed
    atom       = cfg.atom
    method     = cfg.method
    mask_json    = cfg.get("mask_json", None)     # optional reliability flags
    tag_mask_csv = cfg.get("tag_mask_csv", None)  # optional tag_labels.csv

    # without either file the output mask is the residue mask only
    extra_mask_dict = _load_extra_mask_dict(Path(mask_json), atom=atom) if mask_json else {}
    tag_mask_dict = _load_tag_mask_dict(Path(tag_mask_csv)) if tag_mask_csv else {}
    reliability_mask_dict = _and_mask_dicts(extra_mask_dict, tag_mask_dict)

    # ---------- data: every structure in ./structures (no split lists) ---------- #
    dm = PDBDataModule(
        pdb_dir=f"{cwd}/structures",
        processed_dir=f"{cwd}/processed_structures",
        batch_size=batch_size,
        num_workers=0,
        overwrite=False,
        in_memory=False,
        crop=False,
        rand_slice=False,
        inference_only=True,
    )
    inference_loader = dm.inference_dataloader()

    # ---------- model --------------------------------------------- #
    model = CSLightning(embedding=embed, atom=atom).to("cuda")

    # reproducibility: fixed RNG, deterministic kernels
    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)  # strict: raises on an op without a deterministic kernel

    # eval() before any forward pass: disables dropout (the only train/eval-dependent layer)
    model.eval()

    # materialise the encoder's lazy layers with one batch, so that load_state_dict sees the real
    # parameter shapes
    sample_batch = next(iter(inference_loader)).to("cuda")
    with torch.no_grad():
        _ = model.forward(sample_batch)

    if str(ckpt_path).endswith(".safetensors"):        # released weights (state_dict only)
        from safetensors.torch import load_file
        state = {"state_dict": load_file(str(ckpt_path), device="cuda")}
    else:                                              # Lightning checkpoint of a training run
        state = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    missing, unexpected = model.load_state_dict(state["state_dict"], strict=True)
    if missing or unexpected:
        print("State-dict mismatch: missing=", missing, "  unexpected=", unexpected)

    # ---------- inference -------------------------------------------------- #
    output_dict = {"pred": {}, "mask": {}}  # keyed by PDB code (batch.id = structure file name)

    with torch.inference_mode():
        for cur_batch in inference_loader:
            cur_batch = cur_batch.to("cuda")
            pdb_indices = cur_batch.id
            pred, mask = model.forward(cur_batch)
            for i, pdb_index in enumerate(pdb_indices):
                pos_mask = mask[i]

                # AND with the flags / tag mask; an entry not covered is not gated
                ext_mask = reliability_mask_dict.get(str(pdb_index).lower())
                if ext_mask is not None:
                    n = min(len(pos_mask), len(ext_mask))
                    gated = torch.zeros_like(pos_mask)
                    gated[:n] = pos_mask[:n] & ext_mask[:n].to(pos_mask.device)
                    pos_mask = gated

                output_dict["pred"][pdb_index] = pred[i].detach().cpu().numpy().tolist()
                output_dict["mask"][pdb_index] = pos_mask.detach().cpu().numpy().tolist()

    output_path = f"predictions_{atom}_{method}.json"
    with open(output_path, "w") as f:
        json.dump(output_dict, f, indent=4)
    print(f"Saved predictions and masks to {output_path}")


# ------------------------------------------------------------------ #
if __name__ == "__main__":
    cli()
