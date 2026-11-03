#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train SLAE-ERROR: a per-residue 15N chemical-shift error regressor.

Predicts the error of the external SLAE-CS 10-seed ensemble; it has no chemical-shift head of its own.

- Error head (`var_head`): a ShiftNet (input projection, depth-wise conv, TinyContext transformer,
  `depth` ResidualFF/GEGLU blocks, linear output) on the SLAE residue embedding, output mu_err.
- Target: a precomputed residual per residue, read from cfg.oof_json (merge_oof_v3.py) for all three
  splits: K-fold out-of-fold residuals for train, residuals of the 10 full-data models for val/test
  (see cs_oof_data.py). Loss: Huber, delta = 1.0.
- predict_abs_error=true (released model): the target is |residual|; mu_err is clamped to
  [0, mu_err_clamp] and the output bias starts at the train mean |residual| (bias_init).
  Otherwise the signed residual, symmetric clamp, bias 0.0.
- Optional extra inputs, concatenated to the 128-d embedding: the per-residue ensemble std
  (use_std_feature) and the 10 seeds' predictions, sorted and centred on the ensemble mean
  (use_sorted_features). Released model: both on (128 + 1 + 10 = 139 input channels).
- uncertainty_mode: "joint" (encoder from `ckpt`, optionally warm-started from a trained SLAE-CS
  checkpoint, cfg.pretrained_cs_ckpt) or "finetune" (pretrained_cs_ckpt required). Encoder lr =
  lr_last, error-head lr = lr_err.
- freeze_encoder=true: encoder requires_grad_(False) and kept in eval() (see train()); applied in
  cli() after the lazy-layer rebuild of model.encoder.
- Checkpoints: top-3 by val epoch Huber loss; with +ckpt_by_spearman=true also top-3 by val epoch
  Spearman(mu_err, residual), which selected the released model.

Run: see docs/reproduce.md, SLAE-ERROR.
"""

# ------------------------------------------------------------------ #
#  Imports                                                           #
# ------------------------------------------------------------------ #
import os, warnings
from pathlib import Path
from typing import Tuple, Dict
import shutil

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import callbacks as plcb
from scipy.stats import spearmanr
import numpy as np
from omegaconf import DictConfig, OmegaConf
from flash.core.optimizers import LinearWarmupCosineAnnealingLR

# ---------- SLAE specific ------------------------------------ #
from slae_cs.model.shiftnet import ShiftNet
from SLAE.datasets.datamodule import PDBDataModule
from SLAE.features.graph_featurizer import ProteinGraphFeaturizer
import hydra

# ---------- helpers (error_head/cs_*.py) ---------------------- #
from cs_shift_data import build_shift_and_id_sets  # only used for id-set filtering, see cli()
from cs_train_helpers import build_encoder_from_files, TopKSplitCheckpoint
from cs_oof_data import (build_oof_resid_dict, get_oof_resid_and_mask,
                          build_oof_std_dict, get_oof_std,
                          build_oof_sorted_dict, get_oof_sorted,
                          SORTED_FIELDS)

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
PAD_VALUE    = 1e-5
BEST_CKPT_DIR = Path(f"{cwd}/best_ckpts")
EMB_DIM      = 128   # residue-embedding width (matches ShiftNet emb_dim)
# Initial output bias of var_head for predict_abs_error=True. Training passes bias_init = train mean
# |OOF residual| from cli() (1.2501 for the released model); this constant is only the fallback when
# bias_init is not given (rebuilding a model for inference, where the bias is loaded from the
# checkpoint).
MAE_BIAS_INIT = 1.235

# ------------------------------------------------------------------ #
#  LightningModule                                                   #
# ------------------------------------------------------------------ #
class CSLightning(pl.LightningModule):
    def __init__(self, ckpt: Path,
                 lr_last: float = 3e-5,
                 reinit_encoder_weights=False, embedding="residue_embedding", atom: str = "N",
                 # residual targets for all splits (cs_oof_data.build_oof_resid_dict)
                 oof_resid_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = None,
                 # per-residue ensemble std, an extra var_head input channel when use_std_feature
                 oof_std_dict: Dict[str, torch.Tensor] = None,
                 use_std_feature: bool = False,
                 # the 10 seeds' predictions per residue, sorted (seeds are exchangeable) and
                 # centred: 10 extra var_head input channels when use_sorted_features
                 oof_sorted_dict: Dict[str, torch.Tensor] = None,
                 use_sorted_features: bool = False,
                 # output-bias init for predict_abs_error (None -> MAE_BIAS_INIT)
                 bias_init: float = None,
                 predict_uncertainty: bool = True,
                 uncertainty_mode: str = "joint",   # "joint" | "finetune"
                 lr_err: float = 5e-4,
                 mu_reg_weight: float = 1.0,
                 var_head_hidden: int = 256,
                 # number of ResidualFF blocks of var_head (ShiftNet's default is also 4)
                 var_head_depth: int = 4,
                 # True: var_head reads the detached embedding, so no gradient reaches the encoder
                 detach_var_head_input: bool = False,
                 # True: target |oof_resid| instead of the signed residual (see module docstring)
                 predict_abs_error: bool = False,
                 # True: encoder frozen and in eval(); only var_head trains (see apply_freeze_encoder)
                 freeze_encoder: bool = False,
                 # train split only: drop residues with |oof_resid| >= train_max_abs_err from the
                 # loss (None = no cap; not used for the released model)
                 train_max_abs_err: float = None,
                 # True: apply the same cap to val/test (and so to checkpoint selection)
                 apply_err_cap_to_eval: bool = False):
        super().__init__()
        # the per-id dicts are data, not hyperparameters (they would bloat every checkpoint)
        self.save_hyperparameters(ignore=["oof_resid_dict", "oof_std_dict", "oof_sorted_dict"])

        encoder  = build_encoder_from_files(ENCODER_CFG, ckpt,
                                            reinit=reinit_encoder_weights)
        self.embedding = embedding
        self.featuriser = ProteinGraphFeaturizer(radius=8.0, use_atom37=True)

        self.atom = atom
        self.encoder = encoder
        self.oof_resid_dict = oof_resid_dict
        self.oof_std_dict = oof_std_dict if oof_std_dict is not None else {}
        self.use_std_feature = bool(use_std_feature)
        self.oof_sorted_dict = oof_sorted_dict if oof_sorted_dict is not None else {}
        self.use_sorted_features = bool(use_sorted_features)
        self.lr_last = lr_last

        # the freeze itself is applied later (apply_freeze_encoder): cli() rebuilds model.encoder
        # after __init__, which would discard a freeze set here
        self.freeze_encoder = bool(freeze_encoder)
        self.train_max_abs_err = (float(train_max_abs_err)
                                   if train_max_abs_err is not None else None)
        self.apply_err_cap_to_eval = bool(apply_err_cap_to_eval)

        # ---------- error head: mu_err = point estimate of the residual (or |residual|) ---------- #
        self.predict_uncertainty = bool(predict_uncertainty)
        self.uncertainty_mode    = str(uncertainty_mode).lower()
        self.lr_err               = float(lr_err)
        self.mu_reg_weight       = float(mu_reg_weight)
        self.detach_var_head_input = bool(detach_var_head_input)
        self.predict_abs_error   = bool(predict_abs_error)
        # output clamp: a loose safety bound of half the atom's shift window (N: 20 ppm);
        # [0, bound] for |residual|, [-bound, bound] for the signed residual (see forward())
        _window_by_atom = {"N": 40.0, "C": 23.0, "H": 8.0, "CA": 32.0, "CB": 66.0, "HA": 5.0}
        window = _window_by_atom.get(atom, 40.0)
        self.mu_err_clamp = 0.5 * window
        if self.predict_uncertainty:
            h = int(var_head_hidden)
            # var_head: a ShiftNet with its own output bias (init_bias) and analytic_centering=False.
            # ppm_window only sizes ShiftNet's unused .huber attribute; the loss is _error_loss.
            # Input width: 128-d embedding + 1 (std) + 10 (sorted predictions), as enabled.
            var_head_emb_dim = EMB_DIM
            if self.use_std_feature:
                var_head_emb_dim += 1
            if self.use_sorted_features:
                var_head_emb_dim += len(SORTED_FIELDS)
            # bias_init (or MAE_BIAS_INIT) for |residual|; 0.0 for the signed residual
            init_bias = ((float(bias_init) if bias_init is not None else MAE_BIAS_INIT)
                         if self.predict_abs_error else 0.0)
            self.var_head = ShiftNet(emb_dim=var_head_emb_dim, width=h, depth=int(var_head_depth),
                                     ppm_window=(-window / 2, window / 2), bias=init_bias,
                                     analytic_centering=False)
            clamp_str = f"[0, {self.mu_err_clamp}]" if self.predict_abs_error else f"+/-{self.mu_err_clamp}"
            print(f"[ERR-ONLY] var_head (mu_err only, ShiftNet-shaped, bias={init_bias}) enabled | "
                  f"mode={self.uncertainty_mode} | lr_err={self.lr_err} | "
                  f"mu_reg_weight={self.mu_reg_weight} | width={h} | depth={var_head_depth} "
                  f"| mu_err_clamp={clamp_str} | predict_abs_error={self.predict_abs_error} "
                  f"| detach_var_head_input={self.detach_var_head_input} "
                  f"| use_std_feature={self.use_std_feature} | use_sorted_features={self.use_sorted_features} "
                  f"(emb_dim={var_head_emb_dim})")
        else:
            self.var_head = None

        self._buf = {
            "val":  {"mu_err": [], "resid": [], "res_idx": [], "pdb": []},
            "test": {"mu_err": [], "resid": [], "res_idx": [], "pdb": []},
        }

    def _error_loss(self, mu_err, oof_resid, oof_mask):
        """Huber(mu_err, oof_resid), delta 1.0, over the residues in oof_mask. The caller skips
        batches with an empty mask (the mean would be NaN)."""
        return nn.functional.huber_loss(mu_err[oof_mask], oof_resid[oof_mask], delta=1.0)

    @torch.no_grad()
    def _error_abs_diff(self, mu_err, oof_resid, oof_mask):
        """mean(|mu_err - oof_resid|), logged only (not part of the loss)."""
        return (mu_err[oof_mask] - oof_resid[oof_mask]).abs().mean()

    def load_pretrained_cs(self, ckpt_path: str):
        """Load the encoder of a trained SLAE-CS checkpoint (strict=False: its ShiftNet `model.*`
        keys are skipped; var_head keeps its random init)."""
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = sd["state_dict"] if "state_dict" in sd else sd
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"[ERR-ONLY] loaded pretrained CS weights from {ckpt_path}")
        print(f"[ERR-ONLY]   missing={len(missing)} unexpected={len(unexpected)} "
              f"(missing should be only var_head.*, unexpected should be only model.*)")

    def report_trainable_params(self):
        """Print the number of trainable parameters."""
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(f"[ERR-ONLY] trainable params = {n_train:,} / {n_total:,} "
              f"(freeze_encoder={self.freeze_encoder})")

    def apply_freeze_encoder(self):
        """Freeze the encoder if freeze_encoder. Called from cli() after the encoder rebuild."""
        if not self.freeze_encoder:
            return
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()
        print("[FREEZE-ENCODER] encoder frozen (requires_grad=False, eval()) "
              "-- only var_head trains")

    def train(self, mode: bool = True):
        """Keep a frozen encoder in eval() (Lightning calls .train() on the whole module every
        epoch, which would re-enable its dropout). var_head always trains."""
        super().train(mode)
        if mode and self.freeze_encoder:
            self.encoder.eval()
        return self

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        """Same target lookup as training_step; val/test residuals come from the 10 full-data models."""
        split = "val" if dataloader_idx == 0 else "test"
        mu_err, pad_mask = self.forward(batch)
        oof_resid, oof_mask = get_oof_resid_and_mask(self.oof_resid_dict, batch.id, pad_mask)
        if self.predict_abs_error:  # target |resid|, not resid
            oof_resid = oof_resid.abs()

        # the training cap, only if apply_err_cap_to_eval
        if self.train_max_abs_err is not None and self.apply_err_cap_to_eval:
            oof_mask = oof_mask & (oof_resid.abs() < self.train_max_abs_err)

        if oof_mask.sum().item() == 0:
            return

        mu_reg_loss = self._error_loss(mu_err, oof_resid, oof_mask)
        mu_abs_diff = self._error_abs_diff(mu_err, oof_resid, oof_mask)
        self.log_dict({f"{split}_mu_reg_loss": mu_reg_loss,
                       f"{split}_mu_abs_diff": mu_abs_diff},
                      prog_bar=True, on_step=False, on_epoch=True,
                      batch_size=mu_err.size(0), add_dataloader_idx=False)

        b = self._buf[split]
        b["mu_err"].append(mu_err[oof_mask].detach().cpu())
        b["resid"].append(oof_resid[oof_mask].detach().cpu())
        for i, pdb in enumerate(batch.id):
            idx = oof_mask[i].nonzero(as_tuple=False).squeeze(-1)
            if idx.numel():
                b["res_idx"].append(idx.cpu())
                b["pdb"].extend([pdb] * idx.numel())

    @torch.no_grad()
    def on_validation_epoch_end(self):
        log_dir = getattr(self.logger, "save_dir", None) or getattr(self.trainer, "default_root_dir", ".")
        out_dir = Path(log_dir) / "err_history"
        out_dir.mkdir(parents=True, exist_ok=True)

        for split in ("val", "test"):
            b = self._buf[split]
            if not b["mu_err"]:
                continue

            mu_t    = torch.cat(b["mu_err"], 0).float().cpu()
            resid_t = torch.cat(b["resid"], 0).float().cpu()

            # epoch-level metrics over the whole split (not batch-averaged); the top-K trackers in
            # cli() select on epoch_mu_reg_loss and, optionally, epoch_mu_err_spearman
            mu_abs_diff_epoch = float((mu_t - resid_t).abs().mean())
            mu_reg_loss_epoch = float(nn.functional.huber_loss(mu_t, resid_t, delta=1.0))
            # ranking quality: Spearman(mu_err, residual)
            spearman = float(spearmanr(mu_t.numpy(), resid_t.numpy()).correlation)
            if np.isnan(spearman):
                spearman = 0.0

            self.log(f"{split}_epoch_mu_reg_loss", mu_reg_loss_epoch, prog_bar=True, on_step=False, on_epoch=True, add_dataloader_idx=False)
            self.log(f"{split}_epoch_mu_abs_diff", mu_abs_diff_epoch, prog_bar=False, on_step=False, on_epoch=True, add_dataloader_idx=False)
            self.log(f"{split}_epoch_mu_err_spearman", spearman, prog_bar=True, on_step=False, on_epoch=True, add_dataloader_idx=False)
            print(f"[epoch {self.current_epoch:03d}] split={split} "
                  f"epoch_mu_reg_loss={mu_reg_loss_epoch:.4f} epoch_mu_abs_diff={mu_abs_diff_epoch:.4f} "
                  f"epoch_mu_err_spearman={spearman:.4f}")

            # per-residue predictions of this epoch
            idx = torch.cat(b["res_idx"], 0).cpu().numpy()
            pdb = np.array(b["pdb"])
            out_path = out_dir / f"{self.atom}_{split}_epoch{self.current_epoch:03d}_muRegLoss{mu_reg_loss_epoch:.3f}.pt"
            torch.save({"mu_err": mu_t.numpy(), "oof_resid": resid_t.numpy(),
                       "res_idx": idx, "pdb_id": pdb}, out_path)
            print(f"Saved {split} error history to {out_path}")

            for k in b.keys():
                b[k].clear()

    # ---------------- forward -------------------------------------- #
    def forward(self, batch):
        """Returns (mu_err [B, L], residue mask [B, L])."""
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

        # split the flat residue embeddings per protein and pad to [B, L_max, 128]
        res_emb_padded = nn.utils.rnn.pad_sequence(seq_list, batch_first=True,
                                                   padding_value=PAD_VALUE)
        max_len = res_emb_padded.size(1)
        mask = torch.arange(max_len, device=res_emb_padded.device).unsqueeze(0) < lengths.unsqueeze(1)

        mu_err = None
        if self.predict_uncertainty and self.var_head is not None:
            var_input = (res_emb_padded.detach() if self.detach_var_head_input
                         else res_emb_padded)
            # + ensemble std: [B, L, 128] -> [B, L, 129]
            if self.use_std_feature:
                std_feat = get_oof_std(self.oof_std_dict, batch.id, mask)   # [B, L]
                var_input = torch.cat([var_input, std_feat.unsqueeze(-1)], dim=-1)
            # + the 10 sorted, centred seed predictions
            if self.use_sorted_features:
                s_feat = get_oof_sorted(self.oof_sorted_dict, batch.id, mask)   # [B, L, n_sorted]
                var_input = torch.cat([var_input, s_feat], dim=-1)
            raw    = self.var_head(var_input, mask)                     # [B, L] -- ShiftNet.forward
            # |residual| >= 0: one-sided clamp; signed residual: symmetric clamp
            if self.predict_abs_error:
                mu_err = raw.clamp(0.0, self.mu_err_clamp)
            else:
                mu_err = raw.clamp(-self.mu_err_clamp, self.mu_err_clamp)
        return mu_err, mask

    # ---------------- training step -------------------------------- #
    def training_step(self, batch, _ ):
        """Loss: mu_reg_weight * Huber(mu_err, oof_resid)."""
        mu_err, pad_mask = self.forward(batch)
        oof_resid, oof_mask = get_oof_resid_and_mask(self.oof_resid_dict, batch.id, pad_mask)
        if self.predict_abs_error:  # target |resid|, not resid
            oof_resid = oof_resid.abs()

        # optional cap on |residual| (train split only, see __init__)
        if self.train_max_abs_err is not None:
            oof_mask = oof_mask & (oof_resid.abs() < self.train_max_abs_err)

        # skip a batch without targets (an empty mean is NaN); None tells Lightning not to step
        if oof_mask.sum().item() == 0:
            print("WARNING: training batch has zero OOF-covered targets -- skipping.")
            return None

        mu_reg_loss = self._error_loss(mu_err, oof_resid, oof_mask)
        mu_abs_diff = self._error_abs_diff(mu_err, oof_resid, oof_mask)
        loss = self.mu_reg_weight * mu_reg_loss

        self.log_dict({"train_loss": loss, "train_mu_reg_loss": mu_reg_loss,
                       "train_mu_abs_diff": mu_abs_diff},
                      prog_bar=True, on_step=True, on_epoch=True,
                      batch_size=mu_err.size(0))
        return loss

    # ---------------- optimiser + scheduler ------------------------ #
    def configure_optimizers(self):
        # two parameter groups: encoder (lr_last, empty if frozen) and var_head (lr_err)
        groups = []
        enc_params = [p for p in self.encoder.parameters() if p.requires_grad]
        if enc_params:
            groups.append({"params": enc_params, "lr": self.lr_last})
        if self.var_head is not None:
            head_params = [p for p in self.var_head.parameters() if p.requires_grad]
            if head_params:
                groups.append({"params": head_params, "lr": self.lr_err})

        opt = torch.optim.AdamW(groups, weight_decay=1e-4)

        # step-interval schedule: linear warmup over min(100, total/10) steps, then cosine decay
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
@hydra.main(version_base=None, config_path="../configs", config_name="error_head")
def cli(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    pl.seed_everything(cfg.get("seed", 42), workers=True)
    # bit-exact reproducibility: no cudnn auto-tuning, deterministic kernels only (strict)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)

    ckpt = Path(cfg.ckpt)
    batch = cfg.batch
    epochs = cfg.epochs
    workers = cfg.workers
    reinitenc = cfg.reinitenc
    embed = cfg.embed
    fraction = cfg.fraction
    atom = cfg.atom
    run_name = cfg.run_name
    mask_json  = cfg.get("mask_json",  None)      # reliability flags JSON (val/test)
    tag_mask_csv = cfg.get("tag_mask_csv", None)  # tag_labels.csv (all splits); None = disabled
    lr_last = cfg.get("lr_last", 3e-5)

    # ---------- path overrides (default: data/cs/*.json, *.list in the working dir) ---------- #
    train_json = cfg.get("train_json", None)
    val_json   = cfg.get("val_json",   None)
    test_json  = cfg.get("test_json",  None)
    train_list = str(cfg.get("train_list", "train.list"))
    val_list   = str(cfg.get("val_list",   "val.list"))
    test_list  = str(cfg.get("test_list",  "test.list"))
    save_dir   = str(cfg.get("save_dir",  f"{cwd}/logger"))
    ckpt_dir   = str(cfg.get("ckpt_dir",  f"{cwd}/checkpoints"))

    # ---------- error head ---------- #
    predict_uncertainty = bool(cfg.get("predict_uncertainty", False))
    uncertainty_mode    = str(cfg.get("uncertainty_mode", "joint")).lower()
    var_head_hidden     = int(cfg.get("var_head_hidden", 256))
    var_head_depth      = int(cfg.get("var_head_depth", 4))
    pretrained_cs_ckpt  = cfg.get("pretrained_cs_ckpt", None)
    mu_reg_weight       = float(cfg.get("mu_reg_weight", 1.0))
    lr_err              = float(cfg.get("lr_err", 5e-4))
    detach_var_head_input = bool(cfg.get("detach_var_head_input", False))
    predict_abs_error   = bool(cfg.get("predict_abs_error", False))
    freeze_encoder       = bool(cfg.get("freeze_encoder", False))
    train_max_abs_err_cfg = cfg.get("train_max_abs_err", None)
    train_max_abs_err   = float(train_max_abs_err_cfg) if train_max_abs_err_cfg is not None else None
    apply_err_cap_to_eval = bool(cfg.get("apply_err_cap_to_eval", False))

    use_std_feature = bool(cfg.get("use_std_feature", False))
    use_sorted_features = bool(cfg.get("use_sorted_features", False))

    # required: the error-head input table (data/error_head/train_val_test_oof_ensemble_10seed_v3_N.json)
    oof_json = cfg.get("oof_json", None)
    assert predict_uncertainty and oof_json, \
        "chemical_shift_trainer_justerr_oof.py has no chemical-shift loss -- " \
        "predict_uncertainty=true and oof_json=<path to train_oof_N.json> are both required"

    # ---------- id sets of the three splits (ids with at least one valid shift) ---------- #
    _unused_shift_dict, train_ids, val_ids, test_ids = build_shift_and_id_sets(
        atom=atom, train_json=train_json, val_json=val_json, test_json=test_json,
        mask_json=mask_json, tag_mask_csv=tag_mask_csv)
    del _unused_shift_dict

    # residual targets for all three splits (same default paths as cs_shift_data.py)
    real_train_json = Path(train_json) if train_json else (Path(cwd) / "data" / "cs" / "train.json")
    real_val_json   = Path(val_json)   if val_json   else (Path(cwd) / "data" / "cs" / "val.json")
    real_test_json  = Path(test_json)  if test_json  else (Path(cwd) / "data" / "cs" / "test.json")
    oof_resid_dict = build_oof_resid_dict(oof_json,
                                           train_json_path=real_train_json,
                                           val_json_path=real_val_json,
                                           test_json_path=real_test_json,
                                           atom=atom, tag_mask_csv=tag_mask_csv)
    print(f"[OOF] loaded {len(oof_resid_dict)} ids' out-of-fold residuals from {oof_json}")

    # optional input features (empty dict if the field is missing from oof_json)
    oof_std_dict = build_oof_std_dict(oof_json) if use_std_feature else {}
    if use_std_feature:
        print(f"[STD-FEATURE] loaded {len(oof_std_dict)} ids' ensemble std from {oof_json}")
        if not oof_std_dict:
            print("[STD-FEATURE] WARNING: use_std_feature=true but oof_json has no 'N_std' "
                  "field (or is empty) -- var_head will see an all-zero std channel.")

    oof_sorted_dict = build_oof_sorted_dict(oof_json) if use_sorted_features else {}
    if use_sorted_features:
        print(f"[SORTED-FEATURE] loaded {len(oof_sorted_dict)} ids' ensemble sorted values "
              f"({SORTED_FIELDS}) from {oof_json}")
        if not oof_sorted_dict:
            print("[SORTED-FEATURE] WARNING: use_sorted_features=true but oof_json is missing "
                  "one or more sorted fields -- var_head will see all-zero sorted channels.")

    # keep only ids that have a residual target (also sizes the LR schedule correctly)
    oof_ids = set(oof_resid_dict.keys())
    n_train_before = len(train_ids)
    train_ids = train_ids & oof_ids
    val_ids   = val_ids & oof_ids
    test_ids  = test_ids & oof_ids
    print(f"[OOF] train ids: {n_train_before} -> {len(train_ids)} after intersecting with oof_json coverage")
    # initial output bias: mean |OOF residual| over the supervised train residues (1.2501 for the
    # released model)
    _tr = [oof_resid_dict[i] for i in train_ids]
    bias_init = float(torch.cat([r[m].abs() for r, m in _tr if m.any()]).mean())
    print(f"[BIAS-INIT] train mean |OOF resid| = {bias_init:.4f} (used as var_head output bias)")
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

    # ---------- keep only the ids selected above ------------------- #
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
    model = CSLightning(ckpt=ckpt,
                        reinit_encoder_weights=reinitenc,
                        embedding=embed,
                        lr_last = lr_last,
                        atom=atom,
                        oof_resid_dict=oof_resid_dict,
                        oof_std_dict=oof_std_dict,
                        use_std_feature=use_std_feature,
                        oof_sorted_dict=oof_sorted_dict,
                        use_sorted_features=use_sorted_features,
                        bias_init=bias_init,
                        predict_uncertainty=predict_uncertainty,
                        uncertainty_mode=uncertainty_mode,
                        lr_err=lr_err,
                        mu_reg_weight=mu_reg_weight,
                        var_head_hidden=var_head_hidden,
                        var_head_depth=var_head_depth,
                        detach_var_head_input=detach_var_head_input,
                        predict_abs_error=predict_abs_error,
                        freeze_encoder=freeze_encoder,
                        train_max_abs_err=train_max_abs_err,
                        apply_err_cap_to_eval=apply_err_cap_to_eval,
                        ).to("cuda")

    # ---------- lightning trainer --------------------------------- #
    wandb_logger = pl.loggers.WandbLogger(project="SLAE-CS_update",
                                          save_dir=save_dir, name=f"{run_name}")

    # top-3 val checkpoints by epoch_mu_reg_loss (the training objective)
    topk_cb_muRegLoss = TopKSplitCheckpoint(dirpath=ckpt_dir, atom=atom,
                                            metric="epoch_mu_reg_loss", mode="min",
                                            k=3, remove_old=False)
    topk_cbs = [topk_cb_muRegLoss]
    # +ckpt_by_spearman=true: also top-3 by val epoch_mu_err_spearman (the released model was
    # selected this way)
    if bool(cfg.get("ckpt_by_spearman", False)):
        topk_cbs.append(TopKSplitCheckpoint(dirpath=ckpt_dir, atom=atom,
                                            metric="epoch_mu_err_spearman", mode="max",
                                            k=3, remove_old=False))
    callbacks = topk_cbs + [plcb.LearningRateMonitor(logging_interval="step")]

    trainer = pl.Trainer(accelerator="gpu", devices=1, precision=32,
                         max_epochs=epochs,
                         callbacks=callbacks,
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

    # ---------- optional warm start from a trained SLAE-CS encoder; optional freeze ---------- #
    if uncertainty_mode == "finetune":
        assert pretrained_cs_ckpt, \
            "uncertainty_mode='finetune' requires cfg.pretrained_cs_ckpt (a trained CSLightning .ckpt)"
        model.load_pretrained_cs(pretrained_cs_ckpt)
    else:
        # joint: warm start only if pretrained_cs_ckpt is given (else the autoencoder weights)
        if pretrained_cs_ckpt:
            model.load_pretrained_cs(pretrained_cs_ckpt)

    # after the encoder rebuild (required) and the weight load; no-op if freeze_encoder=False
    model.apply_freeze_encoder()
    model.report_trainable_params()
    print(f"[ERR-ONLY] uncertainty_mode={uncertainty_mode} | freeze_encoder={freeze_encoder} | "
          f"pretrained_cs_ckpt={'set' if pretrained_cs_ckpt else 'none'} | "
          f"lr_last={lr_last} (moot if frozen) | lr_err={lr_err}")

    # ---------- train --------------------------------------------- #
    trainer.fit(model, train_loader, [val_loader, test_loader])

    # report the top-3 of each tracker, copy the best val checkpoint to BEST_CKPT_DIR
    BEST_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    copied = []
    for cb in topk_cbs:
        print(f"\n=== Top-3 checkpoints by {cb.metric} ({cb.mode}, computed over full split) ===")

        print("\n[VAL] top-3:")
        for i, (v, pth, ep) in enumerate(cb.top_val, 1):
            print(f"  #{i}: {cb.metric}={v:.6f}  epoch={ep:03d}  path={pth}")

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
