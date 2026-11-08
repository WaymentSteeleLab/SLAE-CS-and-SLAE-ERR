#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train the SLAE-CS shift head on frozen external per-residue embeddings (ESM3, MACE-OFF23 baselines).

Benchmark counterpart of slae_cs/train/chemical_shift_trainer.py: identical shift targets, masks
(flags on val/test, tag mask on all splits), splits, ShiftNet head, loss, validation metrics,
optimizer, LR schedule and Lightning settings. The only change is that the SLAE encoder (+ graph
featuriser) is replaced by a lookup into a packed embedding file (pack_embeddings.py / pack_mace.py),
e.g. ESM3 `seqstruct` (L, 1536). Everything not tied to the encoder is imported from
chemical_shift_trainer.py, so the two cannot drift apart.

Differences, all on the input side / bookkeeping:
  * Embedding normalisation: each dimension is z-scored with the mean/std over all *train* residues
    (ESM3 embeddings are the pre-final-LayerNorm residual stream: per-dim std ~140, a few dims >1000).
    The statistics are stored as buffers in the checkpoint and applied to val/test unchanged.
  * One optimizer param group (the head, at `lr`): there is no encoder, so `lr_last` has nothing to
    act on.
  * The sanity-check validation pass is not written to shift_history/metrics.
  * Per-epoch val/test metrics are written to metrics.json, plus summary.json with the test metrics
    at the best-val epoch (for LR selection / seed statistics).

Run from a working directory made by scripts/make_workdir.py (baselines/README.md), with PYTHONPATH at the
repository root:
    python $R/baselines/embedding_head/embedding_shift_trainer.py hydra.run.dir=. \
        emb_npz=$PWD/esm3_seqstruct.npz emb_name=esm3_seqstruct atom=N lr=3e-4 seed=1 run_name=esm3_seqstruct_seed1 \
        mask_json=$MASK tag_mask_csv=$TAG
"""
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import callbacks as plcb
from torch.utils.data import DataLoader, Dataset
from scipy.stats import spearmanr, pearsonr as pearson
from omegaconf import DictConfig, OmegaConf
from flash.core.optimizers import LinearWarmupCosineAnnealingLR
import hydra

# SLAE-CS trainer: data/mask helpers, CSLightning (training/validation steps), TopKSplitCheckpoint,
# PAD_VALUE. Importing it also sets the same env vars (CUBLAS_WORKSPACE_CONFIG etc.) before any CUDA
# context exists.
from slae_cs.train import chemical_shift_trainer as cst
from slae_cs.model.shiftnet import ShiftNet

# hydra chdirs into outputs/ before cli() runs -- capture the launch dir now
cwd = os.getcwd()

# (ppm_window lo, hi, output bias) per atom -- same values as CSLightning.__init__
SHIFT_WINDOWS = {
    "N":  (100.0, 140.0, 119.0),
    "C":  (163.0, 186.0, 175.0),
    "H":  (4, 12, 8),
    "CA": (40, 72, 56),
    "CB": (14, 80, 38),
    "HA": (2, 7, 4.5),
}


# ------------------------------------------------------------------ #
#  Embeddings                                                        #
# ------------------------------------------------------------------ #
def load_embeddings(npz_path: Path, ids) -> Dict[str, np.ndarray]:
    """pdb_id -> (L, D) float16 for every id in `ids` present in the npz."""
    out = {}
    with np.load(npz_path) as z:
        available = set(z.files)
        for pid in sorted(ids):
            if pid in available:
                out[pid] = z[pid]
    return out


def train_stats(emb: Dict[str, np.ndarray], ids, eps: float = 1e-6):
    """Per-dim mean/std over all residues of the train proteins (float64 sums)."""
    arrs = [emb[p] for p in sorted(ids)]   # fixed order -> bit-identical stats across runs
    d = arrs[0].shape[1]
    s, ss, n = np.zeros(d), np.zeros(d), 0
    for a in arrs:
        a = a.astype(np.float64)
        s += a.sum(0); ss += (a * a).sum(0); n += a.shape[0]
    mean = s / n
    std = np.sqrt(np.maximum(ss / n - mean * mean, 0.0)) + eps
    return torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32), n


class EmbeddingDataset(Dataset):
    def __init__(self, ids, emb: Dict[str, np.ndarray]):
        self.ids = sorted(ids)
        self.emb = emb

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        pid = self.ids[i]
        return pid, self.emb[pid]


@dataclass
class EmbBatch:
    id: List[str]            # pdb ids (same attribute name as the SLAE graph batch)
    emb: torch.Tensor        # [B, L_max, D] raw embeddings, zero-padded
    lengths: torch.Tensor    # [B]


def collate(items) -> EmbBatch:
    ids = [p for p, _ in items]
    arrs = [torch.from_numpy(a.astype(np.float32)) for _, a in items]
    return EmbBatch(ids, nn.utils.rnn.pad_sequence(arrs, batch_first=True),
                    torch.tensor([a.shape[0] for a in arrs]))


# top-K tracker of the SLAE-CS trainer, used as-is
TopKSplitCheckpoint = cst.TopKSplitCheckpoint


# ------------------------------------------------------------------ #
#  LightningModule                                                   #
# ------------------------------------------------------------------ #
class EmbCSLightning(cst.CSLightning):
    """CSLightning with the SLAE encoder swapped for normalised frozen embeddings.

    training_step / validation_step / loss are inherited unchanged from chemical_shift_trainer.
    """
    def __init__(self, shift_dict, emb_mean: torch.Tensor, emb_std: torch.Tensor,
                 lr: float = 1e-4, loss_type: str = "huber", atom: str = "N",
                 analytic_centering: bool = False, history_dir: str = "shift_history"):
        pl.LightningModule.__init__(self)   # skip CSLightning.__init__ (builds the SLAE encoder)
        self.save_hyperparameters(ignore=["shift_dict", "emb_mean", "emb_std"])
        self.register_buffer("emb_mean", emb_mean)
        self.register_buffer("emb_std", emb_std)

        lo, hi, bias = SHIFT_WINDOWS[atom]
        self.model = ShiftNet(emb_dim=emb_mean.numel(), width=256,
                              ppm_window=(lo, hi), bias=bias,
                              analytic_centering=analytic_centering, depth=4)
        print("ShiftNet initialized with emb_dim", emb_mean.numel(), "bias", self.model.bias)

        self.atom = atom
        self.shift_dict = shift_dict
        self.loss_type = loss_type.lower()
        self.lr = lr
        self.history_dir = Path(history_dir)
        self.history: List[dict] = []
        self._buf = {
            "val":  {"pred": [], "true": [], "res_idx": [], "pdb": []},
            "test": {"pred": [], "true": [], "res_idx": [], "pdb": []},
        }

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return EmbBatch(batch.id, batch.emb.to(device), batch.lengths.to(device))

    def forward(self, batch):
        max_len = batch.emb.size(1)
        mask = torch.arange(max_len, device=batch.emb.device).unsqueeze(0) < batch.lengths.unsqueeze(1)
        x = (batch.emb - self.emb_mean) / self.emb_std
        x = x.masked_fill(~mask.unsqueeze(-1), cst.PAD_VALUE)   # same pad value as SLAE-CS
        pred = self.model(x, mask)
        return pred, mask

    # same computation/logging as CSLightning.on_validation_epoch_end, plus
    # sanity-check skip and an in-memory per-epoch metric history
    @torch.no_grad()
    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            for b in self._buf.values():
                for k in b:
                    b[k].clear()
            return
        self.history_dir.mkdir(parents=True, exist_ok=True)

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
            spearman = float(spearmanr(pred_t.numpy(), true_t.numpy()).correlation)
            if np.isnan(spearman):
                spearman = 0.0

            self.log(f"{split}_epoch_RMSE", rmse, prog_bar=True, on_step=False, on_epoch=True, add_dataloader_idx=False)
            self.log(f"{split}_epoch_MAE", mae, prog_bar=False, on_step=False, on_epoch=True, add_dataloader_idx=False)
            self.log(f"{split}_epoch_Pearson", corr, prog_bar=False, on_step=False, on_epoch=True, add_dataloader_idx=False)
            self.log(f"{split}_epoch_Spearman", spearman, prog_bar=True, on_step=False, on_epoch=True, add_dataloader_idx=False)

            print(f"[epoch {self.current_epoch:03d}] split={split} epoch_RMSE={rmse:.4f} epoch_MAE={mae:.4f} pearson={corr:.4f} spearman={spearman:.4f}")
            self.history.append({"epoch": self.current_epoch, "split": split, "RMSE": rmse,
                                 "MAE": mae, "Pearson": corr, "Spearman": spearman,
                                 "n": int(true_t.numel())})

            idx = torch.cat(b["res_idx"], 0).cpu().numpy()
            pdb = np.array(b["pdb"])
            out_path = self.history_dir / f"{self.atom}_{split}_epoch{self.current_epoch:03d}_corr{corr:.3f}_rmse{rmse:.3f}.pt"
            torch.save({"pred": pred_t.numpy(), "true": true_t.numpy(), "res_idx": idx, "pdb_id": pdb}, out_path)

            for k in b.keys():
                b[k].clear()

    # same optimizer/schedule as SLAE-CS, minus the (absent) encoder param group
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
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


def best_epoch_summary(history: List[dict]) -> dict:
    """Best-val-RMSE epoch and the val/test metrics at that same epoch."""
    by = {(h["epoch"], h["split"]): h for h in history}
    val = [h for h in history if h["split"] == "val"]
    if not val:
        return {}
    best = min(val, key=lambda h: h["RMSE"])
    e = best["epoch"]
    out = {"best_epoch": e, "n_epochs": len(val)}
    for split in ("val", "test"):
        h = by.get((e, split))
        if h:
            for k in ("RMSE", "MAE", "Pearson", "Spearman", "n"):
                out[f"{split}_{k}"] = h[k]
    return out


# ------------------------------------------------------------------ #
#  CLI                                                               #
# ------------------------------------------------------------------ #
@hydra.main(version_base=None, config_path="../configs", config_name="embedding_head")
def cli(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    pl.seed_everything(cfg.seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)

    atom = cfg.atom
    batch = cfg.batch
    out_dir = Path(cfg.get("out_dir") or f"{cwd}/run_output")
    ckpt_dir = out_dir / "checkpoints"
    history_dir = out_dir / "shift_history"

    # ---------- targets & ID sets: SLAE-CS code and paths ---------- #
    shift_dict, train_ids, val_ids, test_ids = cst.build_shift_and_id_sets(
        atom=atom,
        train_json=cfg.train_json or f"{cwd}/data/cs/train.json",
        val_json=cfg.val_json or f"{cwd}/data/cs/val.json",
        test_json=cfg.test_json or f"{cwd}/data/cs/test.json",
        mask_json=cfg.mask_json, tag_mask_csv=cfg.tag_mask_csv)

    # as in SLAE-CS training: keep ids that are both in the split .list and in the split's shift JSON
    def read_list(name):
        with open(Path(cwd) / name) as f:
            return {l.strip().lower() for l in f if l.strip()}
    train_ids &= read_list(cfg.train_list)
    val_ids   &= read_list(cfg.val_list)
    test_ids  &= read_list(cfg.test_list)

    # ---------- embeddings ------------------------------------------ #
    emb = load_embeddings(Path(cfg.emb_npz), train_ids | val_ids | test_ids)
    for name, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
        miss = ids - emb.keys()
        print(f"{name}: {len(ids) - len(miss)}/{len(ids)} proteins have embeddings"
              + (f" (missing e.g. {sorted(miss)[:5]})" if miss else ""))
        ids -= miss
    for pid in train_ids | val_ids | test_ids:   # embedding rows must align with shift positions
        n_shift = len(shift_dict[pid][0])
        assert emb[pid].shape[0] == n_shift, (pid, emb[pid].shape, n_shift)

    for name, a, b in [("train-val", train_ids, val_ids),
                       ("train-test", train_ids, test_ids),
                       ("val-test", val_ids, test_ids)]:
        o = a & b
        print(f"ID overlap ({name}):", len(o))
        assert not o, f"Leakage {name}! examples: {list(o)[:5]}"

    emb_mean, emb_std, n_res = train_stats(emb, train_ids)
    print(f"z-score stats from {n_res} train residues: "
          f"std median {emb_std.median():.2f}, max {emb_std.max():.2f}")

    def loader(ids, shuffle):
        return DataLoader(EmbeddingDataset(ids, emb), batch_size=batch, shuffle=shuffle,
                          num_workers=cfg.workers, collate_fn=collate)
    train_loader = loader(train_ids, True)
    val_loader   = loader(val_ids, False)
    test_loader  = loader(test_ids, False)

    # ---------- model / trainer (SLAE-CS settings) --------------------- #
    model = EmbCSLightning(shift_dict=shift_dict, emb_mean=emb_mean, emb_std=emb_std,
                           lr=cfg.lr, loss_type=cfg.loss, atom=atom,
                           analytic_centering=bool(cfg.analytic_centering),
                           history_dir=str(history_dir))

    wandb_logger = pl.loggers.WandbLogger(project=cfg.wandb_project,
                                          save_dir=str(out_dir), name=cfg.run_name)
    topk_cb = TopKSplitCheckpoint(dirpath=str(ckpt_dir), atom=atom,
                                  metric="epoch_RMSE", mode="min", k=3, remove_old=False)
    topk_cb_spearman = TopKSplitCheckpoint(dirpath=str(ckpt_dir), atom=atom,
                                           metric="epoch_Spearman", mode="max", k=3, remove_old=False)

    trainer = pl.Trainer(accelerator="gpu", devices=1, precision=32,
                         max_epochs=cfg.epochs,
                         callbacks=[topk_cb, topk_cb_spearman,
                                    plcb.LearningRateMonitor(logging_interval="step")],
                         log_every_n_steps=10,
                         logger=wandb_logger,
                         gradient_clip_val=1.0)
    trainer.fit(model, train_loader, [val_loader, test_loader])

    # ---------- outputs ---------------------------------------------- #
    best_dir = out_dir / "best_ckpts"
    best_dir.mkdir(parents=True, exist_ok=True)
    for cb in (topk_cb, topk_cb_spearman):
        print(f"\n=== Top-3 checkpoints by {cb.metric} ({cb.mode}) ===")
        for i, (v, pth, ep) in enumerate(cb.top_val, 1):
            print(f"  #{i}: {cb.metric}={v:.6f}  epoch={ep:03d}  path={pth}")
        if cb.top_val:
            shutil.copy2(cb.top_val[0][1], best_dir / Path(cb.top_val[0][1]).name)

    summary = {"run_name": cfg.run_name, "emb_name": cfg.emb_name, "atom": atom,
               "lr": float(cfg.lr), "seed": int(cfg.seed),
               "n_train": len(train_ids), "n_val": len(val_ids), "n_test": len(test_ids),
               **best_epoch_summary(model.history)}
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(model.history, f, indent=1)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("\n=== summary ===\n" + json.dumps(summary, indent=1))
    try:
        wandb_logger.experiment.summary.update(summary)
    except Exception:
        pass


if __name__ == "__main__":
    cli()
