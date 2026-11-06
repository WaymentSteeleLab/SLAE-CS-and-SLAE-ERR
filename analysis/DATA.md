# Data and external dependencies

This page lists the data and third-party code used by the analysis notebooks, and where each file comes from. Neither is stored in this repository: the datasets are on Hugging Face and Zenodo, and third-party code is cloned. The commands below download everything into `data/` and `external/`. The rest of the page describes each dataset (sections 1–5), the external tools (section 6) and known caveats (Notes).

```
data/        datasets (not in git; download below)
external/    third-party code (not in git; download below)
```

Run every notebook and script from the folder that contains it. Figures are written to a `Figs/` folder next to each notebook.

```bash
python scripts/download_analysis_data.py --parts figures   # data/: Zenodo archives + Hugging Face dataset
python scripts/setup_external.py                           # external/: pinned clones of makeshift, UCBShift, 3Dn
```

Each archive is checked against `scripts/zenodo_files.csv` before unpacking; `--verify` also checks the sha256 of every unpacked file.

## Where each file is

| Short name | Source | Size (compressed / unpacked) | Contents |
|---|---|---|---|
| HF | Hugging Face dataset `hws-lab/SLAE-CS-rBMRB` | | split jsons, AF2 models of the 5,445 rBMRB entries (archives, unpacked), tag labels; linked from `hf_data/` as listed in `scripts/hf_data_map.csv` |
| `analysis` | Zenodo `analysis_intermediates.tar.gz` | see `scripts/zenodo_files.csv` | `data/analysis/` (predictions, features, caches) and the inputs of SI Fig S22 |
| `md` | Zenodo `md_1us_trajectories.tar.gz` | | `data/md_1us/` |
| `processed` | Zenodo `rBMRB_processed.tar.gz` | | re-referenced BMRB tables in `data/rBMRB/` |
| `entries` | Zenodo `rBMRB_bmrb_entries.tar.gz` | | NMR-STAR files of the 10,048 entries that pass the first filters |
| 3rd party | original distribution | | not redistributed |
| regenerate | a notebook of this repository | | written when the notebook runs; also included in the archive named in the table |

## 1. Curated BMRB dataset: `data/rBMRB/`

Chemical shifts curated from the BMRB, re-referenced with PANAV and LACS (NOREF = no re-referencing; `rBMRB-BASELINE` on Hugging Face). `_v1` / `_v2` in file names are stages of the curation (`curation/README.md`).

| Path | Content | Size | Source | Used by |
|---|---|---|---|---|
| `cluster_30p/{PANAV,LACS,NOREF}/test.json` | test split (413 proteins), backbone shifts and masks | 9–19 MB each | HF | almost every figure |
| `cluster_30p/{PANAV,LACS,NOREF}/val.json` | validation split | 10–20 MB each | HF | SI Fig S2 |
| `cluster_30p/{PANAV,LACS,NOREF}/train.json` | training split | 113 / 130 / 224 MB | HF | SI Fig S2 |
| `bmrb_cs_data_withoffset_panav_v2.json` | all entries with PANAV offsets | 292 MB | `processed` | Fig 2b, 2f |
| `bmrb_cs_data_withoffset_lacs_v2.json` | all entries with LACS offsets | 170 MB | `processed` | Fig 2b, 2f |
| `bmrb_cs_data_withoffset_noref_v1.json` | all entries, no re-referencing | 175 MB | `processed` | Fig 3d |
| `bmrb_cs_data_withoffset_{lacs,panav}_v1.json` | before matching LACS and PANAV; used for the inherent-noise decomposition | 171 / 329 MB | `processed` | Fig 3b, 3c, SI S6, S7 |
| `bmrb_cs_data_withoffset_panav_v2_flags_only.json` | PANAV per-atom flags | 37 MB | `processed` | SI Fig S2 |
| `tag_labels.csv` | expression-tag and linker residue labels | 1.6 MB | HF | SI Fig S2 |
| `bmrb_cs_data_raw_stage2_noph_notemp.pkl` | parsed BMRB entries with pH and temperature | 402 MB | `processed` | feature pipeline |
| `bmrb_entries/<bmrID>/<bmrID>_*.str` | NMR-STAR files | 2.2 GB | `entries` | Fig 3b, 3c (pair filtering) |

## 2. Structures: `data/rBMRB/cluster_30p/structures_split_filtered/`

AlphaFold2 models (`<entry>.pdb`, pLDDT in the B-factor column) in `train/`, `val/`, `test/` (4,647 / 385 / 413 files; 867 MB). Source: HF.
PyMOL scenes (`*.pml`) and B-factor PDBs are not stored: the structure notebooks write them next to themselves.
Experimental PDB entries (2RU6, 6V5D) are downloaded from RCSB into `pdb_raw/` by the Fig 6g and SI S22 notebooks.


## 3. Model predictions: `data/analysis/`

| Path | Content | Size | Source | Used by |
|---|---|---|---|---|
| `Fig2/results/predictions_N_seed_ensemble_10seed_{panav,lacs,noref,noref_whole}.json`, `test_*.json`, `seeds_results/`, `mcseeds_results/`, `train_fraction_*` | 10-seed ensembles of the three re-referencing models, per-seed and MC-dropout runs, training-fraction runs | 293 MB | `analysis` (`test_*.json`: HF) | Fig 2c–g, SI S4 |
| `Fig4/results/predictions_N_seed_ensemble_10seed{,_esm3,_esm3_seqonly,_mace,_mpnn_ft}.json` | 10-seed ensembles of SLAE-CS and the four baseline encoders on the PANAV test set | 201 MB | `analysis` | Fig 4–6, SI S5, S8–S12, S16, S19–S21 |
| `Fig5/predictions_N_ensemble_10seed_sorted_centered_err.json` | out-of-fold predicted \|error\| of SLAE-ERR | 4.9 MB | `analysis` | Fig 6b, 6e–h, SI S17–S21 |

## 4. Per-residue features of the test set: `data/analysis/Fig4/`

413 proteins, 49,171 residues. Written by `analysis/shared/feature_pipeline/`, in this order: `dssp_calculation…` → `sasa_exposure…` → `extract_foldseek_3di_af2` → `extract_3dn_per_residue` → `compute_geom_features.py` → `run_talosn_batch` → (Dyna-1, run separately) → `merge_features.py` → `add_plddt_ph_temp.py`.
All from `analysis` unless noted.

| Path | Feature | Method | Size |
|---|---|---|---|
| `dssp_results/per_residue_records.json` | secondary structure | DSSP 4.6.1 (conda-forge `mkdssp`) on the AF2 models | 5.5 MB |
| `sasa_results/per_residue_records.json` | DSSP + amide exposure (N+H SASA > 0 Å²) | Biopython SASA | 4 MB |
| `tdi_results/foldseek_3di_per_residue.json` | Foldseek 3Di state | Foldseek `structureto3didescriptor` | 3.6 MB |
| `tdn_results/3dn_per_residue.json` | 3Dn state | 3Dn (Petti et al. 2025) | 4.6 MB |
| `geom_results/geom_features_af2.csv` | dihedrals, amide H-bond, ring current, heavy-atom packing | `compute_geom_features.py` (H-bond and ring-current algorithms from UCBShift) | 18 MB |
| `p_exchange_results/p_exchange/<entry>-Dyna1.csv` | µs–ms exchange probability | Dyna-1 on the AF2 models (run separately) | 2 MB |
| `Talos-N_results/output_new/<bmrID>/predS2.tab` | order parameter S² | TALOS-N via makeshift, from the experimental shifts; 50915_1/_2 and 4293_1/_2 were split into separate inputs | 364 MB |
| `Talos-N_results/inputs/` | TALOS-N input files | `run_talosn_batch.ipynb` | 130 MB |
| `Talos-N_results/talosn_data/` | TALOS-N reference database, only needed to rerun TALOS-N | **3rd party**: TALOS-N distribution | 402 MB |
| `all_features/per_residue_records_merged.json` | merge of the above | `merge_features.py` | 44 MB |
| `all_features/per_residue_records_full.json` | + pH, temperature, pLDDT | `add_plddt_ph_temp.py` | 47 MB |
| `dssp_results/dssp8_test_filtered.json` | 8-state DSSP | helix-capping analysis | 0.6 MB |

## 5. Molecular dynamics and analysis caches

| Path | Content | Size | Source | Used by |
|---|---|---|---|---|
| `data/md_1us/<entry>/frame0.pdb … frame1000.pdb` | 1-µs MD trajectories of the 48 selected proteins, one frame per ns | 7.5 GB | `md` | Fig 5c, 5d, 5f, 5g, SI S9–S11 |
| `data/analysis/Fig4/per_protein_analysis/md_1us_combined_48_justs2_justdihedral_panav10seedens.json` | per-residue S² and dihedral statistics from MD | 1.2 MB | `analysis` | Fig 5c, 5g, 5h, SI S9, S10 |
| `data/analysis/Fig4/per_protein_analysis/{persist90_ss_burial_simple.csv, packN8_cache_combined_48.json, ss_error_attribution_data_panav10seedens.json}` | caches of Fig 5 notebooks | < 1 MB each | `analysis` / regenerate | Fig 5c, 5g, 5h |
| `…/abnormal_coil/md_dssp8_phipsi_gated_allframes_panav10seedens.npz` | per-frame DSSP (8-state) and φ/ψ of the gated residues of the 48 MD proteins (`Fig5/compute_md_dssp8_phipsi_gated_allframes.py`) | 40 MB | `analysis` / regenerate | Fig 5d, 5f, SI S11 |
| `…/abnormal_coil/hk2010_fig6a_wpd_datasets.csv` | region boundaries of the Ramachandran plot of Hollingsworth & Karplus 2010, Fig 6a, digitized with WebPlotDigitizer | 11 kB | `analysis` | Fig 5d, 5f, SI S11 |
| `…/abnormal_coil/{md_phipsi_perframe_coil_persist50_panav10seedens.npz, hk2010_region_labels_coil_persist50_panav10seedens.npz}` | per-frame φ/ψ of residues with ≥ 50 % coil persistence (`Fig5/get_persist_50.ipynb`); their region labels (cache of `Fig5/coil_rama_error_map.ipynb`) | 7.9 / 2.0 MB | `analysis` / regenerate | Fig 5d, 5f, SI S11 |
| `data/analysis/Fig4/{group_pairwise,be_sweep_bonferroni_results}_*.csv` | statistics tables of Fig 5a/5e and the SI S8 per-model notebooks | < 0.1 MB | `analysis` / regenerate | Fig 5a, 5e, SI S8 |
| `data/analysis/Fig3/inherent_test/{decomposition,decomposition_auto_only,decomposition_no_othernoise}.json` | inherent-noise pairs: all / after the automatic filters / after the automatic and manual filters | 0.4 MB | `analysis` / regenerate | Fig 3, Tables S2–S3 |
| `data/analysis/Fig1/random20_af2_structures/output/` | UCBShift predictions for 20 random structures | 0.7 MB | `analysis` | SI S1 |

## 6. External code and tools

| Path / command | Source | Used by |
|---|---|---|
| `external/makeshift` | github.com/WaymentSteeleLab/makeshift | Fig 1b–d, Fig 5c, TALOS-N pipeline, `curation/` |
| `external/CSpred-ucbshift` | github.com/THGLab/CSpred | `compute_geom_features.py` |
| `external/3Dn` | `3Dn_example/` of github.com/spetti/structure_comparison | `extract_3dn_per_residue.ipynb` |
| `foldseek` on `PATH` | Foldseek | `extract_foldseek_3di_af2.ipynb`, `curation/` |
| `mmseqs` on `PATH` | MMseqs2 | `curation/` |
| `mkdssp` on `PATH` | DSSP 4.6.1 (in `environment.yml`) | DSSP notebook, SI S12 |
| Dyna-1 | run separately; outputs in `p_exchange_results/` | exchange feature |
| PyMOL | renders the `*.pml` scenes written by the structure notebooks | structure panels |

The commits are in `scripts/setup_external.py`.

## Notes

1. **Test set: 413 proteins.** All analyses use `cluster_30p/PANAV/test.json`, and the per-residue features cover these 413 proteins.
2. **DSSP.** `dssp_results/per_residue_records.json` (413 proteins, 49,171 residues) matches `sasa_results/per_residue_records.json` residue by residue. `mkdssp` 4.6.1 needs `LIBCIFPP_DATA_DIR=$CONDA_PREFIX/share/libcifpp` if it cannot find its dictionaries.
