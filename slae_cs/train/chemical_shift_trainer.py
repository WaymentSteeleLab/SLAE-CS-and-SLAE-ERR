#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train SLAE-CS: a per-residue chemical-shift regressor (ShiftNet head) on the SLAE encoder.

- Encoder config: SLAE/configs/encoder/protein_encoder.yaml; encoder weights (including the lazy
  layers) come from the pretrained SLAE checkpoint `ckpt`.
- Structures are featurised on the fly by PDBDataModule (./structures, cache in
  ./processed_structures); shift labels come from data/cs/{train,val,test}.json.
- Labels are masked with the tag/linker labels (all splits) and the reliability flags (val/test).
- Checkpoints: top-3 by val epoch RMSE and by val epoch Spearman; the best of each is copied to
  ./best_ckpts.

Run from the repository root (hydra config configs/slae_cs_train.yaml; paths absolute, hydra changes
the working directory), e.g.:
    PYTHONPATH=$PWD python slae_cs/train/chemical_shift_trainer.py ckpt=$PWD/autoencoder.ckpt mask_json=<flags.json> \
        tag_mask_csv=<tag_labels.csv> seed=<s> lr=3e-5 lr_last=3e-4
"""

# ------------------------------------------------------------------ #
#  Imports                                                           #
# ------------------------------------------------------------------ #
import os, math, json, warnings
from pathlib import Path
from typing import List, Tuple, Dict
import shutil

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import callbacks as plcb
from scipy.stats import spearmanr, pearsonr as pearson
import numpy as np
from omegaconf import DictConfig, OmegaConf
from flash.core.optimizers import LinearWarmupCosineAnnealingLR
from pytorch_lightning.callbacks import Callback

# ---------- SLAE specific ------------------------------------ #
from SLAE.model.encoder import ProteinEncoder
from slae_cs.model.shiftnet import ShiftNet
from SLAE.datasets.datamodule import PDBDataModule
from SLAE.features.graph_featurizer import ProteinGraphFeaturizer
import hydra

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
ENCODER_CFG  = Path(f"{cwd}/SLAE/configs/encoder/protein_encoder.yaml")
TRAIN_JSON   = Path(f"{cwd}/data/cs/train.json")
VAL_JSON     = Path(f"{cwd}/data/cs/val.json")
TEST_JSON    = Path(f"{cwd}/data/cs/test.json")
PAD_VALUE    = 1e-5
BEST_CKPT_DIR = Path(f"{cwd}/best_ckpts")

# ------------------------------------------------------------------ #
#  Shift-JSON helpers                                                #
# ------------------------------------------------------------------ #
def _parse_shift_list(lst: List, sequence: str, atom: str = "N") -> Tuple[torch.Tensor, torch.Tensor]:
    """Shift values and validity mask. Missing / non-finite values, and atoms the residue does
    not have (N/H of proline, CB of glycine), are invalid (value 0.0, mask False)."""
    vals, mask = [], []
    for i, x in enumerate(lst):
        residue = sequence[i]
        bad = False
        if x is None:
            bad = True
        elif isinstance(x, str):
            xs = x.strip().lower()
            if xs in {"none", "null", "nan", "inf", "+inf", "-inf"}:
                bad = True
            else:
                try:
                    x = float(x)
                except Exception:
                    bad = True
        if not bad and isinstance(x, (int, float)):
            if not math.isfinite(float(x)):
                bad = True

        if atom == "N" or atom == "H":
            if residue == "P":  # Proline has no amide N or H
                bad = True

        if atom == "CB":
            if residue == "G":  # Glycine has no CB
                 bad = True

        if bad:
            vals.append(0.0); mask.append(False)
        else:
            vals.append(float(x)); mask.append(True)

    return (torch.tensor(vals, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.bool))

# Per-residue reliability mask from the flags JSON ("{atom}_flag", e.g. "N_flag"). A flag is True
# where the shift is unreliable; it is inverted here so that, like every other mask in this file,
# True = keep. Keyed like _json_to_dict (entry_ID, lower-cased, "bmr" prefix stripped).
def _load_extra_mask_dict(mask_json_path: Path, atom: str = "N") -> Dict[str, torch.Tensor]:
    with open(mask_json_path) as f:
        rbmrb = json.load(f)

    mask_key = f"{atom}_flag"
    out = {}
    for ref_id, mask_lst in rbmrb[mask_key].items():
        pdb_id = rbmrb["entry_ID"][ref_id].lower().removeprefix("bmr")
        out[pdb_id] = torch.tensor([not bool(m) for m in mask_lst], dtype=torch.bool)
    return out


# Per-residue keep mask from the tag/linker labels (tag_labels.csv, one label string per entry).
# Label chars: 'A' = native residue, 'P' = proline (native, no N/H), 'T' = expression-tag /
# cleavage-site residue, 't' = flank / linker residue. Tag residues may carry measured shifts but
# are not native protein, so only 'A'/'P' are kept. Keyed like _load_extra_mask_dict.
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


# AND several pdb_id -> bool-mask dicts (reliability flags, tag mask). A pdb_id missing from one
# source is not gated by that source.
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


def _json_to_dict(json_path: Path, atom: str = "N",
                   extra_mask_dict: Dict[str, torch.Tensor] = None) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    with open(json_path) as f:
        rbmrb = json.load(f)

    out = {}
    for ref_id, shift_lst in rbmrb[atom].items():
        if all(x is None or (isinstance(x, str) and x.lower() == "none") or (isinstance(x, str) and x.lower() == "null") for x in shift_lst):
            continue                                          # skip all-None rows
        sequence = rbmrb["sequence"][ref_id]
        shift, value_mask = _parse_shift_list(shift_lst, sequence, atom=atom)

        pdb_id = rbmrb["entry_ID"][ref_id].lower().removeprefix("bmr")

        # AND with the per-residue mask (train: tag mask; val/test: tag mask & flags). Residues
        # beyond the mask's length are dropped; an entry without a mask is not gated.
        if extra_mask_dict is not None:
            ext_mask = extra_mask_dict.get(pdb_id)
            if ext_mask is None:
                pass
            else:
                n = min(len(value_mask), len(ext_mask))
                gated = torch.zeros_like(value_mask)
                gated[:n] = value_mask[:n] & ext_mask[:n]
                value_mask = gated

        out[pdb_id] = (torch.tensor(shift, dtype=torch.float32),
                        torch.tensor(value_mask, dtype=torch.bool))
    return out

# pdb_id is already lower-case and removes "BMR" prefix for consistency with PDBDataModule
def build_shift_and_id_sets(atom: str = "N",
                            train_json=None, val_json=None, test_json=None,
                            mask_json=None, tag_mask_csv=None, flags_on_test: bool = True):
    # flags_on_test=False: the test split gets only the tag mask, like train. Used by the K-fold
    # runs, whose "test" split is the held-out fold of train.json.
    train_json = TRAIN_JSON if train_json is None else Path(train_json)
    val_json   = VAL_JSON   if val_json   is None else Path(val_json)
    test_json  = TEST_JSON  if test_json  is None else Path(test_json)
    # reliability flags: val/test only
    extra_mask_dict = _load_extra_mask_dict(mask_json, atom=atom)

    # tag/linker mask: every split
    tag_mask_dict = _load_tag_mask_dict(Path(tag_mask_csv)) if tag_mask_csv else {}

    train_mask = tag_mask_dict if tag_mask_dict else None
    val_test_mask = _and_mask_dicts(extra_mask_dict, tag_mask_dict)

    shift_train = _json_to_dict(train_json, atom=atom, extra_mask_dict=train_mask)
    shift_val   = _json_to_dict(val_json, atom=atom, extra_mask_dict=val_test_mask)
    shift_test  = _json_to_dict(test_json, atom=atom,
                                extra_mask_dict=val_test_mask if flags_on_test else train_mask)
    # one lookup table for all splits
    shift_dict  = {**shift_train, **shift_val, **shift_test}
    # range of the target values (non-zero entries)
    all_shifts = [v[0] for v in shift_dict.values()]
    all_shifts = np.concatenate(all_shifts)
    all_shifts = all_shifts[all_shifts != 0.0]

    print(f"Min shift for {atom}: {min(all_shifts)}")
    print(f"Max shift for {atom}: {max(all_shifts)}")
    print(f"Mean shift for {atom}: {np.mean(all_shifts)}")
    return shift_dict, set(shift_train.keys()), set(shift_val.keys()), set(shift_test.keys())

# ------------------------------------------------------------------ #
#  Misc utilities                                                    #
# ------------------------------------------------------------------ #
def _to_plain_dict(cfg_node) -> dict:
    """Flatten an OmegaConf node to a plain Python dict (recursive)."""
    return OmegaConf.to_container(cfg_node, resolve=True, structured_config_mode=False)

def init_weights_slae(m: nn.Module):
    """
    • Linear / Conv layers  →  Kaiming-Uniform (fan-in)
    • LayerNorm / BatchNorm →  weight=1, bias=0
    • Embedding             →  Normal(0, 1/√d)
    """
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.constant_(m.weight, 1.)
        nn.init.constant_(m.bias, 0.)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0., std=1./math.sqrt(m.embedding_dim))
    return m

# ------------------------------------------------------------------ #
#  Encoder / decoder builders                                        #
# ------------------------------------------------------------------ #
def build_encoder_from_files(cfg_path: Path, ckpt_path: Path,
                             map_location="cpu", batch=None,
                             reinit=False) -> ProteinEncoder:
    """Instantiate ProteinEncoder and (optionally) load weights from ckpt."""
    full_cfg  = OmegaConf.load(cfg_path)
    enc_cfg   = _to_plain_dict(full_cfg)
    enc_cfg.pop("_target_", None)
    encoder = ProteinEncoder(**enc_cfg).to("cuda")

    if reinit:
        print("+++ Re-initialising ProteinEncoder weights +++")
        return init_weights_slae(encoder)

    ckpt      = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    if "state_dict" in ckpt:
        enc_state = {k.replace("encoder.", ""): v
                     for k, v in ckpt["state_dict"].items()
                     if k.startswith("encoder.")}
    else:
        enc_state = ckpt["encoder"]
    # materialise lazy layers if a dummy batch is provided
    if batch is not None:
        with torch.no_grad():
            _ = encoder(batch)

    missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
    if missing or unexpected:
        print("State-dict mismatch: missing=", missing, "  unexpected=", unexpected)
    encoder.eval()
    return encoder

# ------------------------------------------------------------------ #
#  Callback: top-K checkpointing per split (epoch-level metric)       #
# ------------------------------------------------------------------ #
class TopKSplitCheckpoint(Callback):
    """
    Keep the top-K val checkpoints by an epoch-level metric computed over
    concatenated preds/targets (not batch-averaged) in
    CSLightning.on_validation_epoch_end. The test value is only tracked (best
    value and epoch, for logging); it never selects or saves a checkpoint.
    Saves filenames with the metric value embedded.

    `metric` is the suffix logged as f"{split}_{metric}" (e.g. "epoch_RMSE" or
    "epoch_Spearman"), `mode` is "min" or "max". cli() runs two instances:
    epoch_RMSE (min) and epoch_Spearman (max).

    NOTE: remove_old=False means it will keep all saved checkpoints on disk
    (but will only track top-K in memory / wandb summary).
    """
    def __init__(self, dirpath: str, atom: str, metric: str = "epoch_RMSE",
                 mode: str = "min", k: int = 3, remove_old: bool = False):
        super().__init__()
        assert mode in ("min", "max"), f"mode must be 'min' or 'max', got {mode!r}"
        self.dirpath = Path(dirpath)
        self.atom = atom
        self.metric = metric
        self.mode = mode
        self.k = int(k)
        self.remove_old = bool(remove_old)
        self.dirpath.mkdir(parents=True, exist_ok=True)

        self.top_val: List[Tuple[float, str, int]] = []
        # best test (value, epoch), logged only; test never selects or saves a checkpoint
        self.best_test: Tuple[float, int] = None

    @staticmethod
    def _to_float(x):
        if x is None:
            return None
        if hasattr(x, "detach"):
            return float(x.detach().cpu())
        return float(x)

    def _is_worse_than_current_worst(self, value: float, current_worst: float) -> bool:
        """True if `value` should NOT replace the current worst kept entry."""
        return value >= current_worst if self.mode == "min" else value <= current_worst

    def _maybe_insert(self, trainer, split: str, value: float, epoch: int):
        # called for split="val" only
        arr = self.top_val

        if len(arr) >= self.k:
            current_worst = (max(arr, key=lambda t: t[0])[0] if self.mode == "min"
                              else min(arr, key=lambda t: t[0])[0])
            if self._is_worse_than_current_worst(value, current_worst):
                return

        filename = f"top{self.k}_{split}_{self.atom}_{self.metric}_epoch{epoch:03d}_{value:.4f}.ckpt"
        path = str(self.dirpath / filename)
        trainer.save_checkpoint(path)

        arr.append((value, path, epoch))
        arr.sort(key=lambda t: t[0], reverse=(self.mode == "max"))  # best (arr[0]) first either way

        while len(arr) > self.k:
            worst = arr.pop(-1)
            if self.remove_old:
                try:
                    os.remove(worst[1])
                except Exception:
                    pass

    def on_validation_end(self, trainer, pl_module):
        # on_validation_end runs after the LightningModule's on_validation_epoch_end has logged
        # this epoch's epoch-level metrics (a callback's on_validation_epoch_end runs before it).
        if trainer.sanity_checking:
            return
        epoch = trainer.current_epoch
        val_metric  = self._to_float(trainer.callback_metrics.get(f"val_{self.metric}"))
        test_metric = self._to_float(trainer.callback_metrics.get(f"test_{self.metric}"))

        if val_metric is not None:
            self._maybe_insert(trainer, "val", val_metric, epoch)

        # test: tracked only
        if test_metric is not None:
            is_better = (self.best_test is None or
                         (test_metric < self.best_test[0] if self.mode == "min"
                          else test_metric > self.best_test[0]))
            if is_better:
                self.best_test = (test_metric, epoch)

        if getattr(trainer, "logger", None) and hasattr(trainer.logger, "experiment"):
            try:
                exp = trainer.logger.experiment
                tag = self.metric
                if self.top_val:
                    exp.summary[f"best_val_{tag}"] = self.top_val[0][0]
                    exp.summary[f"best_val_{tag}_ckpt_path"] = self.top_val[0][1]
                    exp.summary[f"top_val_{tag}_ckpts"] = [p for (_, p, _) in self.top_val]
                    exp.summary[f"top_val_{tag}_values"] = [r for (r, _, _) in self.top_val]
                    exp.summary[f"best_val_{tag}_epoch"] = self.top_val[0][2]
                if self.best_test is not None:
                    exp.summary[f"best_test_{tag}"] = self.best_test[0]
                    exp.summary[f"best_test_{tag}_epoch"] = self.best_test[1]
            except Exception:
                pass

# ------------------------------------------------------------------ #
#  Shift-lookup (batch-time)                                         #
# ------------------------------------------------------------------ #
def get_shift_and_mask(shift_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
                       pdb_ids: List[str],
                       pad_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build (shift, mask) tensors aligned with padded residue batch.
    """
    device      = pad_masks.device
    B, L_max    = pad_masks.shape
    shift_pad   = torch.zeros((B, L_max), dtype=torch.float32, device=device)
    shift_mask  = torch.zeros((B, L_max), dtype=torch.bool,  device=device)

    for i, pdb in enumerate(pdb_ids):
        entry = shift_dict.get(pdb.lower())
        if entry is None:
            continue
        shift_1d, mask_1d = entry
        n = min(len(shift_1d), L_max)
        shift_pad[i, :n]  = shift_1d[:n].to(device)
        shift_mask[i, :n] = mask_1d[:n].to(device)

    total_mask = pad_masks & shift_mask
    return shift_pad, total_mask

# ------------------------------------------------------------------ #
#  LightningModule                                                   #
# ------------------------------------------------------------------ #
class CSLightning(pl.LightningModule):
    def __init__(self, ckpt: Path, shift_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
                 lr: float = 1e-3, lr_last: float = 3e-5, loss_type: str = "huber",
                 reinit_encoder_weights=False, embedding="residue_embedding", atom: str = "N",
                 analytic_centering: bool = True):  # cli() passes cfg.analytic_centering (default False)
        super().__init__()
        self.save_hyperparameters(ignore=["shift_dict"])     # for ckpt reproducibility

        encoder  = build_encoder_from_files(ENCODER_CFG, ckpt,
                                            reinit=reinit_encoder_weights)
        self.embedding = embedding
        self.featuriser = ProteinGraphFeaturizer(
            radius=8.0,
            use_atom37=True)

        self.atom = atom

        # ppm window and output-bias prior per atom type
        print("Using ShiftNet")
        if atom == "N":
            min_shift = 100.0
            max_shift = 140.0
            bias = 119.0
        elif atom == "C":
            min_shift = 163.0
            max_shift = 186.0
            bias = 175.0
        elif atom == "H":
            min_shift = 4
            max_shift = 12
            bias = 8
        elif atom == "CA":
            min_shift = 40
            max_shift = 72
            bias =  56
        elif atom == "CB":
            min_shift = 14
            max_shift = 80
            bias = 38
        elif atom == "HA":
            min_shift = 2
            max_shift = 7
            bias = 4.5
        print("Bias", bias)
        # analytic_centering: if True, ShiftNet.loss() removes the per-protein mean residual
        # ("Δref") before the loss. All reported models use False.
        self.model = ShiftNet(emb_dim=128, width=256,
                                ppm_window=(min_shift, max_shift),
                                bias = bias,
                                analytic_centering=analytic_centering, depth=4)
        print("ShiftNet initialized with bias", self.model.bias)

        self.encoder = encoder
        self.shift_dict = shift_dict
        self.loss_type = loss_type.lower()
        self.lr, self.lr_last = lr, lr_last

        # per-epoch buffers for the epoch-level val/test metrics
        self._buf = {
            "val":  {"pred": [], "true": [], "res_idx": [], "pdb": []},
            "test": {"pred": [], "true": [], "res_idx": [], "pdb": []},
        }

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        split = "val" if dataloader_idx == 0 else "test"
        pred, pad_mask = self.forward(batch)
        shift, total_mask = get_shift_and_mask(self.shift_dict, batch.id, pad_mask)

        if total_mask.sum().item() == 0:
            return

        err  = (pred - shift)[total_mask]
        rmse = torch.sqrt((err ** 2).mean())
        mae  = err.abs().mean()
        x = pred[total_mask].detach().cpu().numpy()
        y = shift[total_mask].detach().cpu().numpy()
        rho = spearmanr(x, y).correlation
        if np.isnan(rho):
            rho = 0.0

        self.log_dict({f"{split}_RMSE": rmse,
                    f"{split}_MAE": mae,
                    f"{split}_Spearman": rho},
                    prog_bar=True, on_step=False, on_epoch=True,
                    batch_size=shift.size(0), add_dataloader_idx=False,)

        b = self._buf[split]
        b["pred"].append(pred[total_mask].detach().cpu())
        b["true"].append(shift[total_mask].detach().cpu())
        for i, pdb in enumerate(batch.id):
            idx = total_mask[i].nonzero(as_tuple=False).squeeze(-1)
            if idx.numel():
                b["res_idx"].append(idx.cpu())
                b["pdb"].extend([pdb] * idx.numel())

    @torch.no_grad()
    def on_validation_epoch_end(self):
        log_dir = getattr(self.logger, "save_dir", None) or getattr(self.trainer, "default_root_dir", ".")
        out_dir = Path(log_dir) / "shift_history"
        out_dir.mkdir(parents=True, exist_ok=True)

        for split in ("val", "test"):
            b = self._buf[split]
            if not b["pred"]:
                continue

            pred_t = torch.cat(b["pred"], 0).float().cpu()
            true_t = torch.cat(b["true"], 0).float().cpu()

            err = pred_t - true_t
            rmse = torch.sqrt((err * err).mean()).item()
            mae  = err.abs().mean().item()
            corr = float(pearson(pred_t.numpy(), true_t.numpy())[0])
            # over the whole split, not batch-averaged (validation_step's per-batch values are only
            # a running estimate); these epoch_* metrics drive checkpoint selection
            spearman = float(spearmanr(pred_t.numpy(), true_t.numpy()).correlation)
            if np.isnan(spearman):
                spearman = 0.0

            self.log(f"{split}_epoch_RMSE", rmse, prog_bar=True, on_step=False, on_epoch=True, add_dataloader_idx=False)
            self.log(f"{split}_epoch_MAE", mae, prog_bar=False, on_step=False, on_epoch=True, add_dataloader_idx=False)
            self.log(f"{split}_epoch_Pearson", corr, prog_bar=False, on_step=False, on_epoch=True, add_dataloader_idx=False)
            self.log(f"{split}_epoch_Spearman", spearman, prog_bar=True, on_step=False, on_epoch=True, add_dataloader_idx=False)

            print(f"[epoch {self.current_epoch:03d}] split={split} epoch_RMSE={rmse:.4f} epoch_MAE={mae:.4f} pearson={corr:.4f} spearman={spearman:.4f}")

            # per-residue predictions of this epoch
            idx = torch.cat(b["res_idx"], 0).cpu().numpy()
            pdb = np.array(b["pdb"])
            out_path = out_dir / f"{self.atom}_{split}_epoch{self.current_epoch:03d}_corr{corr:.3f}_rmse{rmse:.3f}.pt"
            torch.save({"pred": pred_t.numpy(), "true": true_t.numpy(), "res_idx": idx, "pdb_id": pdb}, out_path)
            print(f"Saved {split} shift history to {out_path}")

            for k in b.keys():
                b[k].clear()

    # ---------------- forward -------------------------------------- #
    def forward(self, batch):
        batch = self.featuriser(batch)

        if self.embedding == "residue_embedding":
            res_embedding = self.encoder(batch)[self.embedding]
        else:  # node_embedding – use N atoms only
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

        # split the flat residue embeddings per protein and pad to [B, L_max, 128]
        res_emb_padded = nn.utils.rnn.pad_sequence(seq_list, batch_first=True,
                                                   padding_value=PAD_VALUE)
        max_len = res_emb_padded.size(1)
        mask = torch.arange(max_len, device=res_emb_padded.device).unsqueeze(0) < lengths.unsqueeze(1)
        pred = self.model(res_emb_padded, mask)
        return pred, mask

    # ---------------- training step -------------------------------- #
    def training_step(self, batch, _ ):
        pred, pad_mask = self.forward(batch)
        shift, total_mask = get_shift_and_mask(self.shift_dict, batch.id, pad_mask)

        # Skip a batch with zero valid targets (validation_step does the same):
        # loss[mask].mean() over an EMPTY tensor is NaN, and that NaN would backprop
        # into every weight. Returning None tells Lightning there's nothing to step on.
        if total_mask.sum().item() == 0:
            print("WARNING: training batch has zero valid shift targets -- skipping.")
            return None

        if self.loss_type == "mse":
            if self.model.analytic_centering:
                denom = total_mask.sum(-1, keepdim=True).clamp(min=1)
                mean_err = ((pred - shift) * total_mask).sum(-1, keepdim=True) / denom
                pred_adj = pred - mean_err
            else:
                pred_adj = pred
            loss = ((pred_adj - shift) ** 2)[total_mask].mean()
        else:                   # huber
            loss  = self.model.loss(pred, shift, total_mask)

        err = (pred - shift)[total_mask]
        rmse = torch.sqrt((err ** 2).mean())
        mae  = err.abs().mean()

        self.log_dict({"train_loss": loss, "train_RMSE": rmse, "train_MAE": mae},
                      prog_bar=True, on_step=True, on_epoch=True,
                      batch_size=shift.size(0))
        return loss

    # ---------------- optimiser + scheduler ------------------------ #
    def configure_optimizers(self):
        # two parameter groups: the encoder at self.lr_last, the ShiftNet head at self.lr.
        opt = torch.optim.AdamW(
            [
                {"params": self.encoder.parameters(), "lr": self.lr_last},
                {"params": self.model.parameters(),   "lr": self.lr},
            ],
            weight_decay=1e-4,
        )

        # step-interval schedule (interval="step" below) sized from the trainer's actual
        # number of optimizer steps: linear warmup over min(100, total/10) steps, then
        # cosine decay to the end of training.
        total_steps  = self.trainer.estimated_stepping_batches
        warmup_steps = min(100, max(1, total_steps // 10))
        sched = LinearWarmupCosineAnnealingLR(opt,
                                              warmup_epochs=warmup_steps,
                                              max_epochs=total_steps,
                                              warmup_start_lr=1e-6)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched,
                                 "interval": "step",
                                 "name": "lr"}}

# ------------------------------------------------------------------ #
#  CLI                                                               #
# ------------------------------------------------------------------ #
@hydra.main(version_base=None, config_path="../../configs", config_name="slae_cs_train")
def cli(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    pl.seed_everything(cfg.get("seed", 42), workers=True)
    # bit-exact reproducibility: no cudnn auto-tuning, deterministic kernels only (strict: an op
    # without a deterministic implementation raises instead of falling back silently)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)

    ckpt = Path(cfg.ckpt)
    batch = cfg.batch
    epochs = cfg.epochs
    workers = cfg.workers
    loss = cfg.loss
    reinitenc = cfg.reinitenc
    embed = cfg.embed
    fraction = cfg.fraction
    atom = cfg.atom
    run_name = cfg.run_name
    mask_json  = cfg.get("mask_json",  None)      # reliability flags JSON (val/test)
    tag_mask_csv = cfg.get("tag_mask_csv", None)  # tag_labels.csv (all splits); None = disabled
    lr = cfg.get("lr", 5e-4)
    lr_last = cfg.get("lr_last", 3e-5)

    # ---------- path overrides (K-fold runs: per-fold JSONs, lists, output dirs) ---------- #
    train_json = cfg.get("train_json", None)
    val_json   = cfg.get("val_json",   None)
    test_json  = cfg.get("test_json",  None)
    train_list = str(cfg.get("train_list", "train.list"))
    val_list   = str(cfg.get("val_list",   "val.list"))
    test_list  = str(cfg.get("test_list",  "test.list"))
    save_dir   = str(cfg.get("save_dir",  f"{cwd}/logger"))
    ckpt_dir   = str(cfg.get("ckpt_dir",  f"{cwd}/checkpoints"))

    # ---------- load JSONs & build ID sets ------------------------ #
    shift_dict, train_ids, val_ids, test_ids = build_shift_and_id_sets(
        atom=atom, train_json=train_json, val_json=val_json, test_json=test_json,
        mask_json=mask_json, tag_mask_csv=tag_mask_csv,
        flags_on_test=bool(cfg.get("flags_on_test", True)))

    print(len(train_ids))
    print(len(val_ids))
    print(len(test_ids))
    # ---------- data: PDBDataModule ------------------------------- #
    dm = PDBDataModule(
        pdb_dir=f"{cwd}/structures",
        processed_dir=f"{cwd}/processed_structures",
        batch_size=batch,
        dataset_fraction=fraction,
        overwrite=False,
        in_memory=False,
        crop=False,
        rand_slice=False,
        train_list=train_list,
        val_list=val_list,
        test_list=test_list,
    )
    dm.setup(stage="fit")

    # ---------- keep only the ids that have shift labels ---------- #
    raw_train_loader = dm.train_dataloader()
    raw_val_loader   = dm.val_dataloader()
    raw_test_loader  = dm.test_dataloader()

    from torch.utils.data import Subset, DataLoader
    def _filtered_loader(raw_loader, allowed_ids, shuffle):
        raw_ds = raw_loader.dataset
        keep_idx = [i for i in range(len(raw_ds))
                    if getattr(raw_ds[i], "id", "").lower() in allowed_ids]
        filt_ds = Subset(raw_ds, keep_idx)
        return DataLoader(
            filt_ds,
            batch_size=batch,
            shuffle=shuffle,
            num_workers=workers,
            collate_fn=raw_loader.collate_fn,
            pin_memory=getattr(raw_loader, "pin_memory", False),
        )

    train_loader = _filtered_loader(raw_train_loader, train_ids, shuffle=True)
    val_loader   = _filtered_loader(raw_val_loader,   val_ids,   shuffle=False)
    test_loader  = _filtered_loader(raw_test_loader,  test_ids,  shuffle=False)

    # no id may appear in two splits
    def ids_from_loader(loader):
        subset, raw_ds = loader.dataset, loader.dataset.dataset
        return {raw_ds[i].id.lower() for i in subset.indices}

    train_seen = ids_from_loader(train_loader)
    val_seen   = ids_from_loader(val_loader)
    test_seen  = ids_from_loader(test_loader)
    for name, a, b in [("train-val", train_seen, val_seen),
                    ("train-test", train_seen, test_seen),
                    ("val-test", val_seen, test_seen)]:
        o = a & b
        print(f"ID overlap ({name}):", len(o))
        assert not o, f"Leakage {name}! examples: {list(o)[:5]}"

    # ---------- model --------------------------------------------- #
    # default False (the value used for every reported model)
    analytic_centering = bool(cfg.get("analytic_centering", False))
    model = CSLightning(ckpt=ckpt,
                        shift_dict=shift_dict,
                        loss_type=loss,
                        reinit_encoder_weights=reinitenc,
                        embedding=embed,
                        lr = lr,
                        lr_last = lr_last,
                        atom=atom,
                        analytic_centering=analytic_centering).to("cuda")

    # ---------- lightning trainer --------------------------------- #
    wandb_logger = pl.loggers.WandbLogger(project="SLAE-CS",
                                          save_dir=save_dir, name=f"{run_name}")

    topk_cb = TopKSplitCheckpoint(dirpath=ckpt_dir, atom=atom,
                                  metric="epoch_RMSE", mode="min",
                                  k=3, remove_old=False)
    topk_cb_spearman = TopKSplitCheckpoint(dirpath=ckpt_dir, atom=atom,
                                           metric="epoch_Spearman", mode="max",
                                           k=3, remove_old=False)

    trainer = pl.Trainer(accelerator="gpu", devices=1, precision=32,
                         max_epochs=epochs,
                         callbacks=[topk_cb, topk_cb_spearman,
                                    plcb.LearningRateMonitor(logging_interval="step")],
                         log_every_n_steps=10,
                         logger=wandb_logger,
                         gradient_clip_val=1.0)

    # ---------- materialise the encoder's lazy layers on one batch, then load the weights ---------- #
    dm.setup(stage="lazy_init")
    batch = next(iter(dm.val_dataloader())).to("cuda")
    batch_feat = model.featuriser(batch).to("cuda")
    model.encoder = build_encoder_from_files(ENCODER_CFG, ckpt,
                                             batch=batch_feat).to("cuda")
    del batch, batch_feat

    # ---------- train --------------------------------------------- #
    trainer.fit(model, train_loader, [val_loader, test_loader])

    # report / copy / summarise for both trackers (RMSE and Spearman).
    BEST_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    copied = []
    for cb in (topk_cb, topk_cb_spearman):
        print(f"\n=== Top-3 checkpoints by {cb.metric} ({cb.mode}, computed over full split) ===")

        print("\n[VAL] top-3:")
        for i, (v, pth, ep) in enumerate(cb.top_val, 1):
            print(f"  #{i}: {cb.metric}={v:.6f}  epoch={ep:03d}  path={pth}")

        # Copy the best-val checkpoint into BEST_CKPT_DIR
        if cb.top_val:
            src = cb.top_val[0][1]
            dst = str(BEST_CKPT_DIR / Path(src).name)
            shutil.copy2(src, dst)
            copied.append(dst)

        # final wandb summary (also written by the callback during training)
        try:
            exp = wandb_logger.experiment
            tag = cb.metric
            if cb.top_val:
                exp.summary[f"best_val_{tag}"] = cb.top_val[0][0]
                exp.summary[f"best_val_{tag}_ckpt_path"] = cb.top_val[0][1]
                exp.summary[f"top_val_{tag}_ckpts"] = [p for (_, p, _) in cb.top_val]
                exp.summary[f"top_val_{tag}_values"] = [r for (r, _, _) in cb.top_val]
        except Exception:
            pass

    print("\n=== Copied best checkpoints to ===")
    for p in copied:
        print(" ", p)
    try:
        wandb_logger.experiment.summary["copied_best_ckpts"] = copied
    except Exception:
        pass

# ------------------------------------------------------------------ #
if __name__ == "__main__":
    cli()