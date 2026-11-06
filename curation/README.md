# rBMRB curation

The notebooks that built the SLAE-CS-rBMRB dataset from a BMRB dump: the entry filters, the re-referencing
(PANAV, LACS, no re-referencing), the train / val / test split, the tag labels, and the larger training set
from the whole BMRB used to train SLAE-CS. 

The outputs of these notebooks are in the Hugging Face dataset (`hws-lab/SLAE-CS-rBMRB`); the inputs are on Zenodo (below).

## Layout

Each notebook starts by changing into its working folder under `data/curation/`. All other paths are relative
to that folder, as in the original runs.

| notebook folder | working folder |
| --- | --- |
| `curation/rBMRB/` | `data/curation/rBMRB/` |
| `curation/whole_BMRB/Screening_8_whole.ipynb` | `data/curation/whole_BMRB/` |
| `curation/whole_BMRB/assign_labels_panav.ipynb` | `data/curation/whole_BMRB/cluster_30p/` |

`python scripts/download_analysis_data.py --parts curation` fills `data/curation/`:

| path | what | used by |
| --- | --- | --- |
| `rBMRB/bmrb_entries/`, `whole_BMRB/bmrb_entries/` | the BMRB dump (NMR-STAR 3 files, 16,991 entries); the same dump in both folders, because the filters move entries out of it | `Screening*.ipynb` |
| `rBMRB/LACS_PANAV_structures/` | AlphaFold2 models of the 6,127 entries kept by PANAV and LACS from old run; 220 file names have the `test_` prefix (the old test set) | `Screening_6_match.ipynb` |
| `rBMRB/cluster_30p_old/result_cluster.tsv` | Foldseek clusters of the 6,127 structures: `foldseek easy-cluster ../LACS_PANAV_structures result tmp --alignment-type 1 --tmscore-threshold 0.3 --min-seq-id 0.3 --cov-mode 0 --threads 8`, run in `cluster_30p_old/` | `Screening_6_match.ipynb`, `train_cluster_noleak.ipynb` |
| `whole_BMRB/foldseek_pool/` | AlphaFold2 models of the 14,966 entries of the whole-BMRB set | `Screening_8_whole.ipynb` |

Under `data/curation/rBMRB/`, `cluster_30p_old/` is the old split and `cluster_30p/` is the released split.
The released split is also `data/rBMRB/cluster_30p/` in the `figures` download, and `rBMRB-PANAV`, `rBMRB-LACS` and `rBMRB-BASELINE` in the Hugging Face dataset.

## Order and counts

Counts as in the original run.

1. `rBMRB/Screening_1.ipynb`: entry filters on the dump. Multi-entity 4,004, no polypeptide 676, no chemical shifts
   184, modified residues 862, non-standard residues in the sequence 616, missing backbone shifts 211,
   solid-state NMR 256, denatured 117, longer than 500 residues 17. **10,048** entries left. Each filter moves
   the rejected entries out of `bmrb_entries/` into a folder named after the filter, so start from a fresh copy
   of the dump.
2. `rBMRB/Screening_2.ipynb`: one record per assigned chemical-shift list (10,542), then pH (−1,162) and
   temperature (−1,542) filters. **7,838** left (`bmrb_cs_data_raw_stage2_noph_notemp.pkl`).
3. `rBMRB/Screening_3_panav.ipynb`: PANAV re-referencing: 6,902 entries
   (`bmrb_cs_data_withoffset_panav_v1.json` and the flag file).
4. `rBMRB/Screening_4_lacs.ipynb`: LACS re-referencing: 6,268 (`…_lacs_v1.json`).
5. `rBMRB/Screening_5_noref.ipynb`: no re-referencing: 7,758 (`…_noref_v1.json`).
6. `rBMRB/Screening_6_match.ipynb` makes the old split. It finds the entries kept by both PANAV and LACS (**6,123**;
   `…_panav_v2.json`, `…_lacs_v2.json`), then splits the Foldseek clusters of the 6,127 structures into train / val /
   test **5,306 / 406 / 415** (`cluster_30p_old/split_output/`). 48 entries chosen from the old test set for MD stay in test.
7. `rBMRB/Screening_7_match.ipynb` makes the released split from the old split. It starts from
   `cluster_30p_old/split_output/`, removes the 4 entries that are not among the 6,123, keeps the test set fixed and
   removes train / val entries homologous to it: train / val / test **4,647 / 385 / 413** (`cluster_30p/`, with the
   PANAV / LACS / NOREF json files).
8. `whole_BMRB/Screening_8_whole.ipynb`: the whole-BMRB set used to train SLAE-CS. The same filters as step 1
   except multi-entity / solid-state / denatured, no re-referencing: **14,966** entries
   (`bmrb_cs_data_withoffset_noref_whole_v1.json`). The structures are `foldseek_pool/` above. Training entries
   homologous to the val / test entries of step 7 (Foldseek clustering, then MMseqs2) are removed:
   **12,054** (`cluster_30p/train.json`, `train.list`).
9. `whole_BMRB/assign_labels_panav.ipynb` (imports `whole_BMRB/assign_labels.py`, identical to
   `rBMRB/assign_labels.py`): tag labels of the whole-BMRB set.
10. `rBMRB/assign_labels_panav.ipynb` (imports `rBMRB/assign_labels.py`): the tag label of each entry of the released
    split (sequence only) → `tag_labels.csv`.

## Tools

- Python: `analysis/environment.yml` (makeshift needs `nmrglue`).
- makeshift: `python scripts/setup_external.py` puts makeshift 0.2.0 (commit `1bee3eb`) in `external/makeshift` and applies `rBMRB/makeshift_panav.patch`.
  With the patch, `reref_panav(df, cona=False)` skips only the CONA fragment scan and still returns the `deviant` and `suspicious` lists used for the PANAV flags.
- Foldseek (commit `8dc75c74`) and MMseqs2 (commit `99a025e6`) on `PATH` as `foldseek` and `mmseqs`.

## Not in this folder

- The AlphaFold2 models were predicted separately (see Methods); they are on Zenodo.
- K-fold folds: `scripts/kfold/make_kfold_splits.py` and `splits/kfold/`.
