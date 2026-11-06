"""
Adds to per_residue_records_merged.json (output of merge_features.py):
  - ph, temperature  (one scalar per entry from the raw rBMRB BMRB pickle, broadcast to all residues)
  - plddt            (per residue, from the B-factor column of the CA atom in the rBMRB test-set PDB files)

Sources:
  PH_TEMP_PKL: ../../../data/rBMRB/bmrb_cs_data_raw_stage2_noph_notemp.pkl
      dict["bmr<entry>"] -> {"deuterated":..., "reference":..., "pH":..., "temperature":..., "chemical shifts":...}
  PDB_DIR:     ../../../data/rBMRB/cluster_30p/structures_split_filtered/test/<entry>.pdb
      AlphaFold models; the B-factor column holds pLDDT, read from each residue's CA atom.

As in merge_features.py, the base data (its output) is treated as complete;
a field is None when the new source lacks that entry or (entry, resnum).
Residue types in the PDB (three-letter -> one-letter) are checked against
the base residue_type (ph/temperature are per-chain scalars and are not checked).

Output: per_residue_records_full.json, same structure as the input
(dict[entry] -> list[dict]) with ph/temperature/plddt added to each record.
"""

import json
import os
import pickle

from merge_features import (
    OUTPUT_PATH as MERGED_INPUT_PATH,
    load_json,
    check_residue_type,
    report_missing_rates,
)

FIG4_DIR = "../../../data/analysis/Fig4"

PH_TEMP_PKL = os.path.join(
    "../../../data/rBMRB",
    "bmrb_cs_data_raw_stage2_noph_notemp.pkl",
)
PDB_DIR = os.path.join(
    "../../../data/rBMRB",
    "cluster_30p", "structures_split_filtered", "test",  
)

OUTPUT_PATH = "../../../data/analysis/Fig4/all_features/per_residue_records_full.json"

# three-letter PDB residue code -> one-letter, for the residue_type consistency check
AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def load_ph_temp(pkl_path):
    """entry -> (ph, temperature); entry looks like '10029_1', the pickle key is 'bmr10029_1'"""
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    lookup = {}
    for bmr_key, rec in raw.items():
        entry = bmr_key[len("bmr"):] if bmr_key.startswith("bmr") else bmr_key
        lookup[entry] = (rec.get("pH"), rec.get("temperature"))
    return lookup


def parse_pdb_ca_plddt_and_resname(path):
    """Return {resnum: (plddt, resname_1letter)} from each residue's CA atom."""
    out = {}
    with open(path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            resname3 = line[17:20].strip()
            resnum = int(line[22:26])
            bfactor = float(line[60:66])
            out[resnum] = (bfactor, AA3TO1.get(resname3))
    return out


def build_plddt_lookup(entries, pdb_dir):
    """(entry, resnum) -> {'plddt': ...}; also returns resname for the residue-type check"""
    lookup = {}
    resname_lookup = {}
    for entry in entries:
        path = os.path.join(pdb_dir, f"{entry}.pdb")
        if not os.path.exists(path):
            continue
        for resnum, (plddt, resname) in parse_pdb_ca_plddt_and_resname(path).items():
            key = (entry, resnum)
            lookup[key] = {"plddt": plddt}
            if resname is not None:
                resname_lookup[key] = resname
    return lookup, resname_lookup


def enrich():
    if not os.path.exists(MERGED_INPUT_PATH):
        raise FileNotFoundError(
            f"{MERGED_INPUT_PATH} not found; run merge_features.py first"
        )
    merged = load_json(MERGED_INPUT_PATH)

    ph_temp_lookup = load_ph_temp(PH_TEMP_PKL)
    plddt_lookup, plddt_resname = build_plddt_lookup(merged.keys(), PDB_DIR)

    mismatches = []
    full = {}
    for entry, records in merged.items():
        ph, temperature = ph_temp_lookup.get(entry, (None, None))
        full_records = []
        for rec in records:
            resnum = rec["resnum"]
            key = (entry, resnum)
            base_residue_type = rec.get("residue_type")

            new_rec = dict(rec)
            new_rec["ph"] = ph
            new_rec["temperature"] = temperature
            new_rec.update(plddt_lookup.get(key, {"plddt": None}))

            check_residue_type(
                mismatches, "pdb", entry, resnum,
                base_residue_type, plddt_resname.get(key),
            )

            full_records.append(new_rec)
        full[entry] = full_records

    with open(OUTPUT_PATH, "w") as f:
        json.dump(full, f, indent=2)

    print(f"Wrote enriched data for {len(full)} entries to {OUTPUT_PATH}")

    if mismatches:
        print(f"WARNING: {len(mismatches)} residue-type mismatches found vs base data:")
        for source_name, entry, resnum, expected, actual in mismatches[:20]:
            print(f"  [{source_name}] entry={entry} resnum={resnum} "
                  f"base={expected!r} vs {source_name}={actual!r}")
        if len(mismatches) > 20:
            print(f"  ... and {len(mismatches) - 20} more")
    else:
        print("Residue-type consistency check passed for all sources that provide residue type.")

    report_missing_rates(full)


if __name__ == "__main__":
    enrich()
