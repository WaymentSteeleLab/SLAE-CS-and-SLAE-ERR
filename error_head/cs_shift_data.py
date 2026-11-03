#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shift-JSON loading and masking helpers of the error head: the same functions as in
slae_cs/train/chemical_shift_trainer.py (without its flags_on_test option). The error head uses them
for the id sets of the splits and, through cs_oof_data.py, for the measured shifts.
"""

import os
import math
import json
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import numpy as np

# ------------------------------------------------------------------ #
#  Paths & constants                                                 #
# ------------------------------------------------------------------ #
cwd = os.getcwd()
TRAIN_JSON = Path(f"{cwd}/data/cs/train.json")
VAL_JSON   = Path(f"{cwd}/data/cs/val.json")
TEST_JSON  = Path(f"{cwd}/data/cs/test.json")


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

# Per-residue reliability mask from the flags JSON ("{atom}_flag" = True where unreliable), inverted
# to True = keep. Keyed like _json_to_dict (entry_ID, lower-cased, "bmr" prefix stripped).
def _load_extra_mask_dict(mask_json_path: Path, atom: str = "N") -> Dict[str, torch.Tensor]:
    with open(mask_json_path) as f:
        rbmrb = json.load(f)

    mask_key = f"{atom}_flag"
    out = {}
    for ref_id, mask_lst in rbmrb[mask_key].items():
        pdb_id = rbmrb["entry_ID"][ref_id].lower().removeprefix("bmr")
        out[pdb_id] = torch.tensor([not bool(m) for m in mask_lst], dtype=torch.bool)
    return out


# Per-residue keep mask from the tag/linker labels (tag_labels.csv): 'A' = native residue,
# 'P' = proline, 'T' = expression-tag / cleavage-site residue, 't' = flank / linker residue.
# Only 'A'/'P' are kept. Keyed like _load_extra_mask_dict.
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
                            mask_json=None, tag_mask_csv=None):
    # JSON paths default to data/cs/{train,val,test}.json
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
    shift_test  = _json_to_dict(test_json, atom=atom, extra_mask_dict=val_test_mask)
    # one lookup table for all splits; print the range of the target values (non-zero entries)
    shift_dict  = {**shift_train, **shift_val, **shift_test}
    all_shifts = [v[0] for v in shift_dict.values()]
    all_shifts = np.concatenate(all_shifts)
    all_shifts = all_shifts[all_shifts != 0.0]

    print(f"Min shift for {atom}: {min(all_shifts)}")
    print(f"Max shift for {atom}: {max(all_shifts)}")
    print(f"Mean shift for {atom}: {np.mean(all_shifts)}")
    return shift_dict, set(shift_train.keys()), set(shift_val.keys()), set(shift_test.keys())

