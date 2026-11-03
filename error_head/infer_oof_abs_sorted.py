#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SLAE-ERR inference on the test split with a trained chemical_shift_trainer_justerr_oof_sorted.py
checkpoint. The architecture options (predict_abs_error, use_std_feature, use_sorted_features,
var_head_*) must be those used in training, or the checkpoint does not load (strict=True).
Writes predictions_{atom}_{method}_err.json: predicted error ("pred"), residue mask ("mask"), and the
target residual with its mask ("oof_resid", "oof_mask"). Always runs at batch size 1.

Run:
    PYTHONPATH=$PWD python error_head/infer_oof_abs_sorted.py ckpt=$PWD/autoencoder.ckpt \
        +trained_ckpt=<model.safetensors or trained .ckpt> \
        oof_json=<train_val_test_oof_ensemble_10seed_v3_N.json> mask_json=... tag_mask_csv=... \
        predict_abs_error=true use_std_feature=true use_sorted_features=true \
        uncertainty_mode=joint method=<tag>
"""
import os, json, warnings
from pathlib import Path

import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import hydra

from SLAE.datasets.datamodule import PDBDataModule
from cs_shift_data import build_shift_and_id_sets
from cs_train_helpers import build_encoder_from_files
from cs_oof_data import (build_oof_resid_dict, get_oof_resid_and_mask,
                          build_oof_std_dict,
                          build_oof_sorted_dict)
from chemical_shift_trainer_justerr_oof_sorted import CSLightning, ENCODER_CFG

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   # deterministic cuBLAS (set before CUDA init)

cwd = os.getcwd()


@hydra.main(version_base=None, config_path="../configs", config_name="error_head")
def cli(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    pl.seed_everything(cfg.get("seed", 42), workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)

    ckpt = Path(cfg.ckpt)                        # autoencoder.ckpt: builds the encoder (weights replaced below)
    trained_ckpt = Path(cfg.trained_ckpt)        # the trained SLAE-ERR checkpoint
    # always 1: with padding, ShiftNet's depth-wise conv (kernel 5, before any masking) mixes pad
    # vectors into the last 2 residues of the shorter protein, so predictions would depend on
    # batch composition
    batch = 1
    if cfg.batch != 1:
        print(f"[INFER] ignoring cfg.batch={cfg.batch} -- inference always runs with batch size 1")
    workers = cfg.workers
    embed = cfg.embed
    atom = cfg.atom
    mask_json  = cfg.get("mask_json",  None)
    tag_mask_csv = cfg.get("tag_mask_csv", None)
    var_head_hidden = int(cfg.get("var_head_hidden", 256))
    var_head_depth  = int(cfg.get("var_head_depth", 4))
    predict_abs_error = bool(cfg.get("predict_abs_error", False))
    use_std_feature = bool(cfg.get("use_std_feature", False))
    use_sorted_features = bool(cfg.get("use_sorted_features", False))
    uncertainty_mode = str(cfg.get("uncertainty_mode", "joint")).lower()

    train_json = cfg.get("train_json", None)
    val_json   = cfg.get("val_json",   None)
    test_json  = cfg.get("test_json",  None)
    train_list = str(cfg.get("train_list", "train.list"))
    val_list   = str(cfg.get("val_list",   "val.list"))
    test_list  = str(cfg.get("test_list",  "test.list"))

    oof_json = cfg.get("oof_json", None)
    assert oof_json, "oof_json=<path to train_oof_N.json> is required"

    # id sets of the splits (only test is used)
    _unused_shift_dict, train_ids, val_ids, test_ids = build_shift_and_id_sets(
        atom=atom, train_json=train_json, val_json=val_json, test_json=test_json,
        mask_json=mask_json, tag_mask_csv=tag_mask_csv)
    del _unused_shift_dict

    real_train_json = Path(train_json) if train_json else (Path(cwd) / "data" / "cs" / "train.json")
    real_val_json   = Path(val_json)   if val_json   else (Path(cwd) / "data" / "cs" / "val.json")
    real_test_json  = Path(test_json)  if test_json  else (Path(cwd) / "data" / "cs" / "test.json")
    oof_resid_dict = build_oof_resid_dict(oof_json,
                                           train_json_path=real_train_json,
                                           val_json_path=real_val_json,
                                           test_json_path=real_test_json,
                                           atom=atom, tag_mask_csv=tag_mask_csv)
    print(f"[INFER] loaded {len(oof_resid_dict)} ids' OOF residuals from {oof_json}")

    oof_std_dict = build_oof_std_dict(oof_json) if use_std_feature else {}
    if use_std_feature:
        print(f"[STD-FEATURE] loaded {len(oof_std_dict)} ids' ensemble std from {oof_json}")

    oof_sorted_dict = build_oof_sorted_dict(oof_json) if use_sorted_features else {}
    if use_sorted_features:
        print(f"[SORTED-FEATURE] loaded {len(oof_sorted_dict)} ids' ensemble sorted values from {oof_json}")
        if not oof_sorted_dict:
            print("[SORTED-FEATURE] WARNING: use_sorted_features=true but oof_json is missing "
                  "one or more sorted fields -- var_head will see all-zero sorted channels.")

    oof_ids = set(oof_resid_dict.keys())
    test_ids = test_ids & oof_ids
    print(f"[INFER] test ids: {len(test_ids)}  (val skipped -- test-only inference)")

    dm = PDBDataModule(
        pdb_dir=f"{cwd}/structures",
        processed_dir=f"{cwd}/processed_structures",
        batch_size=batch,
        dataset_fraction=1.0,
        overwrite=False,
        in_memory=False,
        crop=False,
        rand_slice=False,
        train_list=train_list,
        val_list=val_list,
        test_list=test_list,
    )
    dm.setup(stage="fit")

    raw_test_loader = dm.test_dataloader()

    from torch.utils.data import Subset, DataLoader
    def _filtered_loader(raw_loader, allowed_ids):
        raw_ds = raw_loader.dataset
        keep_idx = [i for i in range(len(raw_ds))
                    if getattr(raw_ds[i], "id", "").lower() in allowed_ids]
        filt_ds = Subset(raw_ds, keep_idx)
        return DataLoader(
            filt_ds, batch_size=batch, shuffle=False, num_workers=workers,
            collate_fn=raw_loader.collate_fn,
            pin_memory=getattr(raw_loader, "pin_memory", False),
        )

    test_loader = _filtered_loader(raw_test_loader, test_ids)

    # ---------- model (architecture must match the trained checkpoint) ---- #
    model = CSLightning(ckpt=ckpt,
                        embedding=embed,
                        atom=atom,
                        oof_resid_dict=oof_resid_dict,
                        oof_std_dict=oof_std_dict,
                        use_std_feature=use_std_feature,
                        oof_sorted_dict=oof_sorted_dict,
                        use_sorted_features=use_sorted_features,
                        predict_uncertainty=True,
                        uncertainty_mode=uncertainty_mode,
                        var_head_hidden=var_head_hidden,
                        var_head_depth=var_head_depth,
                        predict_abs_error=predict_abs_error,
                        ).to("cuda")

    model.eval()

    # ---------- materialise the encoder's lazy layers on one batch (as in training) ---- #
    dm.setup(stage="lazy_init")
    lazy_batch = next(iter(dm.test_dataloader())).to("cuda")
    batch_feat = model.featuriser(lazy_batch).to("cuda")
    model.encoder = build_encoder_from_files(ENCODER_CFG, ckpt, batch=batch_feat).to("cuda")
    del lazy_batch, batch_feat

    # ---------- load the trained weights ----------------- #
    if str(trained_ckpt).endswith(".safetensors"):     # released weights (state_dict only)
        from safetensors.torch import load_file
        state = {"state_dict": load_file(str(trained_ckpt), device="cuda")}
    else:                                              # Lightning checkpoint of a training run
        state = torch.load(trained_ckpt, map_location="cuda", weights_only=False)
    missing, unexpected = model.load_state_dict(state["state_dict"], strict=True)
    if missing or unexpected:
        print("State-dict mismatch: missing=", missing, "  unexpected=", unexpected)

    # ---------- inference --------------- #
    method = str(cfg.get("method", None) or "oof")
    output_dict = {"pred": {}, "mask": {}, "oof_resid": {}, "oof_mask": {}}

    with torch.inference_mode():
        for cur_batch in test_loader:
            cur_batch = cur_batch.to("cuda")
            mu_err, mask = model.forward(cur_batch)
            oof_resid, oof_mask = get_oof_resid_and_mask(oof_resid_dict, cur_batch.id, mask)
            if predict_abs_error:  # target |residual|, as in training
                oof_resid = oof_resid.abs()
            for i, pdb_id in enumerate(cur_batch.id):
                output_dict["pred"][pdb_id]      = mu_err[i].detach().cpu().numpy().tolist()
                output_dict["mask"][pdb_id]      = mask[i].detach().cpu().numpy().tolist()
                output_dict["oof_resid"][pdb_id] = oof_resid[i].detach().cpu().numpy().tolist()
                output_dict["oof_mask"][pdb_id]  = oof_mask[i].detach().cpu().numpy().tolist()

    output_path = f"{cwd}/predictions_{atom}_{method}_err.json"
    with open(output_path, "w") as f:
        json.dump(output_dict, f, indent=4)
    print(f"[INFER] saved {len(output_dict['pred'])} ids' predictions to {output_path}")


if __name__ == "__main__":
    cli()
