#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Residual targets and ensemble input features of the error head, read from the error-head input
table (oof_json, written by merge_oof_v3.py). Used by chemical_shift_trainer_justerr_oof_sorted.py and
infer_oof_abs_sorted.py.

Why out-of-fold: a residual computed by a model on its own training ids (in-sample) is not an honest
estimate of held-out error. In the table, each train id's prediction comes from the K-fold models that
never saw that id (scripts/kfold/make_kfold_splits.py), and each val/test id's prediction from the 10
full-data models, which were trained on the train ids only.

Same (value, mask) dict + batch-lookup pattern as the shift labels of SLAE-CS. Ids missing from oof_json get an
all-False mask, so they do not supervise the error head.
"""
import json
from pathlib import Path
from typing import Dict, Tuple

import torch

from cs_shift_data import _parse_shift_list, _load_tag_mask_dict


def _load_json_index(json_path):
    with open(json_path) as f:
        j = json.load(f)
    idx = {eid.lower().removeprefix("bmr"): k for k, eid in j["entry_ID"].items()}
    return j, idx


def build_oof_resid_dict(oof_json_path: Path,
                          train_json_path: Path = None,
                          val_json_path: Path = None,
                          test_json_path: Path = None,
                          atom: str = "N",
                          tag_mask_csv: Path = None) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    with open(oof_json_path) as f:
        oof = json.load(f)

    sources = [_load_json_index(p) for p in (train_json_path, val_json_path, test_json_path)
               if p is not None]
    tag_mask_dict = _load_tag_mask_dict(Path(tag_mask_csv)) if tag_mask_csv else {}

    out = {}
    for out_key, entry_id in oof["entry_ID"].items():
        pdb_id = entry_id.lower().removeprefix("bmr")

        # the split json (train / val / test) that contains this id
        match = None
        for j, idx in sources:
            true_key = idx.get(pdb_id)
            if true_key is not None:
                match = (j, true_key)
                break
        if match is None:
            continue  # id in oof json isn't in any of the *_json_path args passed
        true_json, true_key = match

        oof_pred = torch.tensor(oof["N"][out_key], dtype=torch.float32)
        oof_mask = torch.tensor(oof["N_mask"][out_key], dtype=torch.bool)

        # measured shift and its validity mask, parsed as in cs_shift_data._json_to_dict
        sequence = true_json["sequence"][true_key]
        true_shift, true_mask = _parse_shift_list(true_json[atom][true_key], sequence, atom=atom)

        # tag/linker mask (tag_labels.csv), applied to every split as for SLAE-CS
        ext_mask = tag_mask_dict.get(pdb_id)

        n = min(len(oof_pred), len(true_shift))
        if ext_mask is not None:
            n = min(n, len(ext_mask))
        resid = oof_pred[:n] - true_shift[:n]     # residual = prediction - measured shift
        mask = oof_mask[:n] & true_mask[:n]
        if ext_mask is not None:
            mask = mask & ext_mask[:n]

        out[pdb_id] = (resid, mask)
    return out


def get_oof_resid_and_mask(oof_resid_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
                            pdb_ids, pad_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Padded [B, L] residuals and mask for a batch (as chemical_shift_trainer.get_shift_and_mask); ids
    missing from oof_resid_dict get an all-False mask row."""
    device   = pad_masks.device
    B, L_max = pad_masks.shape
    resid_pad  = torch.zeros((B, L_max), dtype=torch.float32, device=device)
    resid_mask = torch.zeros((B, L_max), dtype=torch.bool,  device=device)

    for i, pdb in enumerate(pdb_ids):
        entry = oof_resid_dict.get(pdb.lower())
        if entry is None:
            continue
        resid_1d, mask_1d = entry
        n = min(len(resid_1d), L_max)
        resid_pad[i, :n]  = resid_1d[:n].to(device)
        resid_mask[i, :n] = mask_1d[:n].to(device)

    return resid_pad, resid_mask & pad_masks


# ---------- input feature: per-residue ensemble std ("N_std") ---------- #
def build_oof_std_dict(oof_json_path: Path) -> Dict[str, torch.Tensor]:
    """pdb_id -> per-residue ensemble std. Returns {} if oof_json has no "N_std" field."""
    with open(oof_json_path) as f:
        oof = json.load(f)
    if "N_std" not in oof:
        return {}
    out = {}
    for out_key, entry_id in oof["entry_ID"].items():
        pdb_id = entry_id.lower().removeprefix("bmr")
        out[pdb_id] = torch.tensor(oof["N_std"][out_key], dtype=torch.float32)
    return out


def get_oof_std(oof_std_dict: Dict[str, torch.Tensor], pdb_ids, pad_masks: torch.Tensor) -> torch.Tensor:
    """Padded [B, L] std for a batch; missing ids and padded positions get 0.0."""
    device   = pad_masks.device
    B, L_max = pad_masks.shape
    std_pad = torch.zeros((B, L_max), dtype=torch.float32, device=device)

    for i, pdb in enumerate(pdb_ids):
        entry = oof_std_dict.get(pdb.lower())
        if entry is None:
            continue
        n = min(len(entry), L_max)
        std_pad[i, :n] = entry[:n].to(device)

    return std_pad


# ---------- input feature: the 10 seeds' predictions, sorted and centred ("N_sort00".."N_sort09") ---------- #
# Sorted because the seeds are exchangeable (no meaningful fixed order); merge_oof_v3.py stores each
# sorted value minus the ensemble mean.
SORTED_FIELDS = [f"N_sort{i:02d}" for i in range(10)]


def build_oof_sorted_dict(oof_json_path: Path) -> Dict[str, torch.Tensor]:
    """pdb_id -> [n_res, 10] tensor. Returns {} if oof_json lacks any of SORTED_FIELDS."""
    with open(oof_json_path) as f:
        oof = json.load(f)
    if not all(field in oof for field in SORTED_FIELDS):
        return {}
    out = {}
    for out_key, entry_id in oof["entry_ID"].items():
        pdb_id = entry_id.lower().removeprefix("bmr")
        cols = [torch.tensor(oof[field][out_key], dtype=torch.float32) for field in SORTED_FIELDS]
        out[pdb_id] = torch.stack(cols, dim=-1)  # [n_res, 10]
    return out


def get_oof_sorted(oof_sorted_dict: Dict[str, torch.Tensor], pdb_ids,
                    pad_masks: torch.Tensor) -> torch.Tensor:
    """Padded [B, L, 10] sorted features for a batch; missing ids and padded positions get 0.0."""
    device   = pad_masks.device
    B, L_max = pad_masks.shape
    n_s = len(SORTED_FIELDS)
    s_pad = torch.zeros((B, L_max, n_s), dtype=torch.float32, device=device)

    for i, pdb in enumerate(pdb_ids):
        entry = oof_sorted_dict.get(pdb.lower())
        if entry is None:
            continue
        n = min(entry.shape[0], L_max)
        s_pad[i, :n, :] = entry[:n].to(device)

    return s_pad
