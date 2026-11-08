#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inference for ProteinMPNN fine-tuning checkpoints (mpnn_launcher.py runs)
=========================================================================
Counterpart of slae_cs/inference/chemical_shift_predictor.py for the
ProteinMPNN-encoder runs. Writes the same predictions_{atom}_{run_tag}_{split}.json
format ({"pred": {pdb: [...]}, "mask": {pdb: [...]}}) so the same ensemble
merge (scripts/ensemble/merge_seed_predictions.py) and scoring apply unchanged.

The model is exactly the trained one: importing mpnn_launcher applies the same
encoder swap used in training (build_encoder_from_files -> ProteinMPNN encoder,
CSLightning.forward -> SLAE batch -> ProteinMPNN h_V -> ShiftNet), then the
checkpoint's full state_dict (encoder + ShiftNet) is loaded with strict=True.

Structures come from the same processed_structures cache the training used
(PDBDataModule, test/val .list), so the inputs are bit-identical to the
ones the val/test metrics were logged on during training.

Same inference rules as the SLAE-CS predictor:
  * batch size 1 (ShiftNet's depth-wise conv would otherwise mix padding into
    the last residues of the shorter protein in a batch);
  * eval() mode (dropout off), deterministic kernels;
  * output mask = residue exists AND flag reliability mask AND tag mask
    (loaded with the training code's own helpers).

Run from a working directory made by scripts/make_workdir.py, PYTHONPATH at the repository root:
    python $R/baselines/proteinmpnn/mpnn_ft_predictor.py --ckpt weights/seed01 --run_tag mpnn_ft_seed01 --split test \
        --mask_json $MASK --tag_mask_csv $TAG --out_dir .
--ckpt takes a released folder, a model.safetensors or a training .ckpt.
"""
import argparse
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file

import mpnn_launcher                        # noqa: F401  (applies the encoder swap)
from slae_cs.train import chemical_shift_trainer as cst
from SLAE.datasets.datamodule import PDBDataModule


def load_weights(path):
    """state_dict and hyper-parameters from a released folder / model.safetensors, or a training .ckpt."""
    path = Path(path)
    st = path / "model.safetensors" if path.is_dir() else path
    if st.suffix == ".safetensors":
        cfg = st.parent / "config.json"
        hp = json.load(open(cfg)).get("hyper_parameters", {}) if cfg.exists() else {}
        return load_file(str(st)), hp
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt["state_dict"], ckpt.get("hyper_parameters", {})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="released folder, model.safetensors or .ckpt")
    p.add_argument("--run_tag", required=True, help="names the output file")
    p.add_argument("--split", action="append", required=True, help="val / test (repeatable)")
    p.add_argument("--mask_json", required=True)
    p.add_argument("--tag_mask_csv", default=None)
    p.add_argument("--out_dir", default=".")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = p.parse_args()

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)

    device = torch.device(args.device)
    sd, hp = load_weights(args.ckpt)
    atom = hp.get("atom", "N")
    model = cst.CSLightning(ckpt=Path("unused.ckpt"), shift_dict={}, atom=atom,
                            loss_type=hp.get("loss_type", "huber"),
                            analytic_centering=hp.get("analytic_centering", False))
    model.load_state_dict(sd, strict=True)
    model = model.to(device).eval()

    mask_dict = cst._and_mask_dicts(
        cst._load_extra_mask_dict(args.mask_json, atom=atom),
        cst._load_tag_mask_dict(Path(args.tag_mask_csv)) if args.tag_mask_csv else {})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    for split in args.split:
        dm = PDBDataModule(pdb_dir=f"{cwd}/structures", processed_dir=f"{cwd}/processed_structures",
                           batch_size=1, overwrite=False, in_memory=False, crop=False, rand_slice=False,
                           train_list="train.list", val_list="val.list", test_list="test.list")
        dm.setup(stage="fit")
        loader = dm.val_dataloader() if split == "val" else dm.test_dataloader()

        out = {"pred": {}, "mask": {}}
        with torch.inference_mode():
            for batch in loader:
                assert len(batch.id) == 1, "inference must run with batch size 1"
                batch = batch.to(device)
                pred, pad = model(batch)
                pid = batch.id[0].lower()
                pos = pad[0].cpu()
                ext = mask_dict.get(pid)
                if ext is not None:
                    n = min(len(pos), len(ext))
                    gated = torch.zeros_like(pos)
                    gated[:n] = pos[:n] & ext[:n]
                    pos = gated
                out["pred"][pid] = pred[0].cpu().numpy().tolist()
                out["mask"][pid] = pos.numpy().tolist()
        path = out_dir / f"predictions_{atom}_{args.run_tag}_{split}.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=4)
        print(f"{Path(args.ckpt).name} [{split}]: {len(out['pred'])} proteins -> {path}")


if __name__ == "__main__":
    main()
