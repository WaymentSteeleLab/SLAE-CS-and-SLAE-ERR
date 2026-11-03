#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Encoder building, weight init and top-K checkpointing for the error head; the same helpers as in
slae_cs/train/chemical_shift_trainer.py.
"""

import os
import math
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback

from SLAE.model.encoder import ProteinEncoder


# ------------------------------------------------------------------ #
#  Misc utilities                                                    #
# ------------------------------------------------------------------ #
def _to_plain_dict(cfg_node) -> dict:
    """Flatten an OmegaConf node to a plain Python dict (recursive)."""
    return OmegaConf.to_container(cfg_node, resolve=True, structured_config_mode=False)

def init_weights_slae(m: nn.Module):
    """
    Linear / Conv layers  ->  Kaiming-Uniform (fan-in)
    LayerNorm / BatchNorm ->  weight=1, bias=0
    Embedding             ->  Normal(0, 1/sqrt(d))
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
#  Encoder builder                                                   #
# ------------------------------------------------------------------ #
def build_encoder_from_files(cfg_path: Path, ckpt_path: Path,
                             map_location="cpu", batch=None,
                             reinit=False) -> ProteinEncoder:
    """Build ProteinEncoder and load the `encoder.*` weights of ckpt (or random init if reinit).
    Pass a featurised batch to materialise the lazy layers before loading."""
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
    Keep the top-K val checkpoints by an epoch-level metric computed over the whole split in
    CSLightning.on_validation_epoch_end. The test value is only tracked (best value and epoch, for
    logging); it never selects or saves a checkpoint. Saves filenames with the metric value embedded.

    `metric` is the suffix logged as f"{split}_{metric}" (error head: "epoch_mu_reg_loss" with
    mode "min", "epoch_mu_err_spearman" with mode "max").

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
                if self.best_test is not None:
                    exp.summary[f"best_test_{tag}"] = self.best_test[0]
                    exp.summary[f"best_test_{tag}_epoch"] = self.best_test[1]
            except Exception:
                pass
