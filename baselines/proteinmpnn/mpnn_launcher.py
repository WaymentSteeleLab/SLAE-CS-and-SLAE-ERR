#!/usr/bin/env python3
"""
ProteinMPNN-encoder fine-tuning launcher.

Wraps the unmodified `chemical_shift_trainer.py` and swaps only the structure
encoder: SLAE's ProteinEncoder -> the encoder half of a pretrained ProteinMPNN
(backbone N/CA/C/O only). Everything else is shared with SLAE-CS: the same
PDBDataModule and processed structures, JSON shift targets and masks, ShiftNet
head (emb_dim=128 matches ProteinMPNN's hidden size), Huber loss, discriminative
LRs (encoder lr_last, head lr), scheduler, seeds and top-K checkpointing. The trainer file
itself is never edited (the swap is done by monkey-patching at import).

Run from a working directory made by scripts/make_workdir.py, PYTHONPATH at the repository root (absolute paths:
hydra changes directory):
  python $R/baselines/proteinmpnn/mpnn_launcher.py hydra.run.dir=. atom=N seed=1 epochs=25 lr=3e-5 lr_last=1e-3 \
      run_name=mpnn_ft_seed1 ckpt=$PWD/unused.ckpt mask_json=$MASK tag_mask_csv=$TAG

  ckpt is required by the config but never opened. reinitenc=true: randomly initialised encoder (control).

Env:
  MPNN_WEIGHTS     ProteinMPNN checkpoint (default external/ProteinMPNN/v_48_020.pt)
  DET_WARN_ONLY=1  downgrade the trainer's strict deterministic mode to warnings,
                   in case a ProteinMPNN op (e.g. gather backward) has no
                   deterministic CUDA kernel in your torch version
"""
import os

import hydra
import torch

# the source trainer (unmodified)
from slae_cs.train import chemical_shift_trainer as T
from mpnn_encoder import DEFAULT_WEIGHTS, ProteinMPNNEncoder, slae_batch_to_mpnn


MPNN_WEIGHTS = os.environ.get("MPNN_WEIGHTS", DEFAULT_WEIGHTS)
_ENCODER = None


def build_mpnn_encoder(cfg_path=None, ckpt_path=None, map_location="cpu",
                       batch=None, reinit=False):
    """Drop-in for T.build_encoder_from_files (SLAE cfg/ckpt are ignored).

    The trainer calls this twice: once in CSLightning.__init__ (with `reinit`) and
    again for the SLAE lazy-layer sanity check (without `reinit`). ProteinMPNN has
    no lazy layers, so the second call returns the same instance; otherwise it
    would silently replace a re-initialised encoder with pretrained weights.
    """
    global _ENCODER
    if _ENCODER is None:
        print(f"+++ ProteinMPNN encoder: {'random init' if reinit else MPNN_WEIGHTS} +++")
        _ENCODER = ProteinMPNNEncoder(MPNN_WEIGHTS, pretrained=not reinit).to("cuda" if torch.cuda.is_available() else "cpu")
    return _ENCODER


def mpnn_forward(self, batch):
    """Replaces CSLightning.forward: SLAE batch -> ProteinMPNN h_V -> ShiftNet."""
    X, mask, pad_mask, residue_idx, chain_enc = slae_batch_to_mpnn(batch)
    h_V = self.encoder(X, mask, residue_idx, chain_enc)            # [B, L, 128]
    h_V = h_V.masked_fill(~pad_mask[..., None], T.PAD_VALUE)
    pred = self.model(h_V, pad_mask)
    return pred, pad_mask


T.build_encoder_from_files = build_mpnn_encoder
T.CSLightning.forward = mpnn_forward

if os.environ.get("DET_WARN_ONLY") == "1":
    _use_det = torch.use_deterministic_algorithms
    T.torch.use_deterministic_algorithms = lambda mode, warn_only=False: _use_det(mode, warn_only=True)


@hydra.main(version_base=None, config_path="../../configs", config_name="slae_cs_train")
def main(cfg):
    print(f"[mpnn_launcher] weights={MPNN_WEIGHTS}  reinitenc={cfg.reinitenc}  "
          f"run_name={cfg.run_name}")
    T.cli(cfg)


if __name__ == "__main__":
    main()
