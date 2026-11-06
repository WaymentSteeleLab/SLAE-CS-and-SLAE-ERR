"""
Merge sasa_results/per_residue_records.json (the complete base data) with:
  - tdn_results/3dn_per_residue.json          (tdn, tdn_idx)
  - tdi_results/foldseek_3di_per_residue.json (tdi)
  - p_exchange_results/p_exchange/*.csv       (p_exchange)
  - geom_results/geom_features_af2.csv        (phi, psi, ... geometric feature columns)
  - Talos-N_results/output_new/*/predS2.tab   (s2, fifth column S2)

The join key is (entry, resnum); entry looks like "10029_1" (BMRB id + suffix).
The sasa data is treated as the complete base: every resnum of every entry is kept,
and a feature is None when its source lacks that (entry, resnum).

Output: all_features/per_residue_records_merged.json, same structure as the input
(dict[entry] -> list[dict]) with the new feature fields appended to each record.
"""

import csv
import json
import os
import re

FIG4_DIR = "../../../data/analysis/Fig4"

SASA_PATH = os.path.join(FIG4_DIR, "sasa_results", "per_residue_records.json")
TDN_PATH = os.path.join(FIG4_DIR, "tdn_results", "3dn_per_residue.json")
TDI_PATH = os.path.join(FIG4_DIR, "tdi_results", "foldseek_3di_per_residue.json")
PEXCHANGE_DIR = os.path.join(FIG4_DIR, "p_exchange_results", "p_exchange")
GEOM_CSV_PATH = os.path.join(FIG4_DIR, "geom_results", "geom_features_af2.csv")
TALOSN_OUTPUT_DIR = os.path.join(FIG4_DIR, "Talos-N_results", "output_new")

OUTPUT_PATH = "../../../data/analysis/Fig4/all_features/per_residue_records_merged.json"

# all columns of geom_features_af2.csv to merge (besides entry/resnum)
GEOM_FEATURE_COLUMNS = [
    "prev_aa", "next_aa", "is_prepro",
    "hb_HN_dist", "hb_HN_cosA", "hb_HN_cosH", "hb_HN_exists", "hb_HN_energy",
    "rc_N", 
    "pack_N8",
    "phi_sin", "phi_cos",
    "psi_sin", "psi_cos",
    "psi_prev_sin", "psi_prev_cos",
    "phi_next_sin", "phi_next_cos",
    "chi1_sin", "chi1_cos",
]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def to_number_or_none(value):
    """Empty strings / NA become None; otherwise try float/int."""
    if value is None:
        return None
    if isinstance(value, float) and value != value:  # NaN
        return None
    s = str(value).strip()
    if s == "" or s.upper() == "NA":
        return None
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except ValueError:
        return s  # non-numeric columns (e.g. prev_aa/next_aa) stay strings


def build_tdn_lookup(tdn_data):
    """(entry, resnum) -> {'tdn': ...}; also returns resname per key for the residue-type check"""
    lookup = {}
    resname_lookup = {}
    for entry, records in tdn_data.items():
        for rec in records:
            key = (entry, rec["resnum"])
            lookup[key] = {"tdn": rec.get("tdn")}
            resname = rec.get("resname")
            if resname is not None:
                resname_lookup[key] = resname
    return lookup, resname_lookup


def build_tdi_lookup(tdi_data):
    """(entry, resnum) -> {'tdi': ...}; also returns resname per key for the residue-type check"""
    lookup = {}
    resname_lookup = {}
    for entry, records in tdi_data.items():
        for rec in records:
            key = (entry, rec["resnum"])
            lookup[key] = {"tdi": rec.get("tdi")}
            resname = rec.get("resname")
            if resname is not None:
                resname_lookup[key] = resname
    return lookup, resname_lookup


def build_pexchange_lookup(pexchange_dir):
    """(entry, resnum) -> {'p_exchange': ...}; entry is parsed from the file name '<entry>-Dyna1.csv'.
    Also returns the residue per key for the residue-type check."""
    lookup = {}
    resname_lookup = {}
    if not os.path.isdir(pexchange_dir):
        return lookup, resname_lookup
    for fname in os.listdir(pexchange_dir):
        if not fname.endswith("-Dyna1.csv"):
            continue
        entry = fname[: -len("-Dyna1.csv")]
        with open(os.path.join(pexchange_dir, fname), newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                resnum = int(row["position"])
                key = (entry, resnum)
                lookup[key] = {
                    "p_exchange": to_number_or_none(row.get("p_exchange"))
                }
                residue = row.get("residue")
                if residue is not None and residue != "":
                    resname_lookup[key] = residue
    return lookup, resname_lookup


def build_geom_lookup(geom_csv_path):
    """(entry, resnum) -> {col: value, ...} for GEOM_FEATURE_COLUMNS"""
    lookup = {}
    if not os.path.exists(geom_csv_path):
        return lookup
    with open(geom_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry = row["entry"]
            resnum = int(row["resnum"])
            lookup[(entry, resnum)] = {
                col: to_number_or_none(row.get(col)) for col in GEOM_FEATURE_COLUMNS
            }
    return lookup


def resolve_talosn_dir(entry, talosn_output_dir):
    """
    entry looks like '10029_1'; a BMRB id can have several entries (e.g. '4293_1'/'4293_2').
    Most TALOS-N folders are named 'bmr<base_id>' (no suffix),
    but when one base_id has several entries the folder carries the suffix, i.e. 'bmr<entry>'.
    Try the folder without the suffix first, then the one with the full entry suffix.
    """
    m = re.match(r"^(\d+)_(\d+)$", entry)
    base_id = m.group(1) if m else entry

    candidate_nosuffix = os.path.join(talosn_output_dir, f"bmr{base_id}")
    if os.path.isdir(candidate_nosuffix):
        return candidate_nosuffix

    candidate_withsuffix = os.path.join(talosn_output_dir, f"bmr{entry}")
    if os.path.isdir(candidate_withsuffix):
        return candidate_withsuffix

    return None


def parse_pred_s2_tab(path):
    """
    Parse predS2.tab and return {resnum: (s2_value, resname)}.
    Format (the VARS line names the columns; S2 is the fifth column, RESNAME the second, used for the residue-type check):
        VARS RESID RESNAME CS_CNT CS_CNT_R2 S2
        FORMAT %4d %1s %2d %2d %8.3f

            2 K  8  3    0.751
    REMARK/VARS/FORMAT lines and blank lines are skipped.
    """
    resnum_to_s2 = {}
    if not os.path.exists(path):
        return resnum_to_s2
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("REMARK") or line.startswith("VARS") or line.startswith("FORMAT"):
                continue
            parts = line.split()
            # RESID RESNAME CS_CNT CS_CNT_R2 S2
            if len(parts) < 5:
                continue
            try:
                resnum = int(parts[0])
                s2 = float(parts[4])
            except ValueError:
                continue
            resnum_to_s2[resnum] = (s2, parts[1])
    return resnum_to_s2


def build_talosn_lookup(sasa_data, talosn_output_dir):
    """(entry, resnum) -> {'s2': ...}; locates each entry's TALOS-N folder and parses predS2.tab.
    Also returns RESNAME per key for the residue-type check."""
    lookup = {}
    resname_lookup = {}
    for entry in sasa_data.keys():
        entry_dir = resolve_talosn_dir(entry, talosn_output_dir)
        if entry_dir is None:
            continue
        pred_s2_path = os.path.join(entry_dir, "predS2.tab")
        resnum_to_s2 = parse_pred_s2_tab(pred_s2_path)
        for resnum, (s2, resname) in resnum_to_s2.items():
            key = (entry, resnum)
            lookup[key] = {"s2": s2}
            resname_lookup[key] = resname
    return lookup, resname_lookup


def report_missing_rates(merged):
    """Print the fraction of None per field after merging (base sasa fields included for reference)."""
    total = sum(len(records) for records in merged.values())
    if total == 0:
        return

    none_counts = {}
    for records in merged.values():
        for rec in records:
            for field, value in rec.items():
                if value is None:
                    none_counts[field] = none_counts.get(field, 0) + 1

    print(f"\nMissing-value report ({total} residues total):")
    all_fields = merged[next(iter(merged))][0].keys() if merged else []
    for field in all_fields:
        count = none_counts.get(field, 0)
        print(f"  {field}: {count}/{total} missing ({count / total:.1%})")


def check_residue_type(mismatches, source, entry, resnum, expected, actual):
    """expected: residue_type from sasa (the base); actual: the residue type reported by another source.
    A mismatch is recorded only when both exist and differ; missing values are skipped."""
    if expected is None or actual is None:
        return
    if str(expected) != str(actual):
        mismatches.append((source, entry, resnum, expected, actual))


def merge():
    sasa_data = load_json(SASA_PATH)
    tdn_lookup, tdn_resname = build_tdn_lookup(load_json(TDN_PATH))
    tdi_lookup, tdi_resname = build_tdi_lookup(load_json(TDI_PATH))
    pexchange_lookup, pexchange_resname = build_pexchange_lookup(PEXCHANGE_DIR)
    geom_lookup = build_geom_lookup(GEOM_CSV_PATH)
    talosn_lookup, talosn_resname = build_talosn_lookup(sasa_data, TALOSN_OUTPUT_DIR)

    # residue-type consistency check, only for sources that report a residue type
    residue_type_sources = [
        ("tdn", tdn_resname),
        ("tdi", tdi_resname),
        ("p_exchange", pexchange_resname),
        ("Talos-N", talosn_resname),
    ]

    geom_none = {col: None for col in GEOM_FEATURE_COLUMNS}

    mismatches = []
    merged = {}
    for entry, records in sasa_data.items():
        merged_records = []
        for rec in records:
            resnum = rec["resnum"]
            key = (entry, resnum)
            base_residue_type = rec.get("residue_type")

            new_rec = dict(rec)  # keep the original fields (complete base data)
            new_rec.update(tdn_lookup.get(key, {"tdn": None}))
            new_rec.update(tdi_lookup.get(key, {"tdi": None}))
            new_rec.update(pexchange_lookup.get(key, {"p_exchange": None}))
            new_rec.update(geom_lookup.get(key, geom_none))
            new_rec.update(talosn_lookup.get(key, {"s2": None}))

            for source_name, resname_lookup in residue_type_sources:
                check_residue_type(
                    mismatches, source_name, entry, resnum,
                    base_residue_type, resname_lookup.get(key),
                )

            merged_records.append(new_rec)
        merged[entry] = merged_records

    with open(OUTPUT_PATH, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"Wrote merged data for {len(merged)} entries to {OUTPUT_PATH}")

    if mismatches:
        print(f"WARNING: {len(mismatches)} residue-type mismatches found vs sasa base data:")
        for source_name, entry, resnum, expected, actual in mismatches[:20]:
            print(f"  [{source_name}] entry={entry} resnum={resnum} "
                  f"base={expected!r} vs {source_name}={actual!r}")
        if len(mismatches) > 20:
            print(f"  ... and {len(mismatches) - 20} more")
    else:
        print("Residue-type consistency check passed for all sources that provide residue type.")

    report_missing_rates(merged)


if __name__ == "__main__":
    merge()
