# SLAE-CS: analysis code

Notebooks and scripts that make the figures of the SLAE-CS paper. `docs/figures.md` (and `MANIFEST.csv`) lists the code of each panel; `DATA.md` lists the data and where each file comes from.

## Layout

```
analysis/
├── Fig1/ … Fig6/              one folder per main figure
├── SI/FigS<N>/                one folder per supplementary figure
└── shared/feature_pipeline/   per-residue features used by Figs 4–6 and the SI
data/                          datasets (not in git)
external/                      third-party code (not in git)
```

## Running

1. Create the environment: `conda env create -f analysis/environment.yml && conda activate slae-cs-analysis`.
   `mkdssp` and `huggingface-hub` come with it; `foldseek` must be on `PATH` for `shared/feature_pipeline/extract_foldseek_3di_af2.ipynb`.
2. From the repository root: `python scripts/download_analysis_data.py --parts figures` and `python scripts/setup_external.py`.
3. Open a notebook from its own folder (all paths are relative to the notebook). Figures are written to `Figs/` next to it.

Order matters only for the steps that produce inputs for others:

- `shared/feature_pipeline/`: DSSP → SASA → Foldseek 3Di → 3Dn → (TALOS-N) → `python compute_geom_features.py` → `python merge_features.py` → `python add_plddt_ph_temp.py`.

Figures 3 and S6, S7:

| Run first | It writes | Needed by |
|---|---|---|
| `Fig3/offset_vs_inherent_1_decomposition.ipynb` | `decomposition.json` | `Fig3/binding_vs_corr_rmse.ipynb` |
| `Fig3/binding_vs_corr_rmse.ipynb` | `decomposition_no_othernoise.json`, `decomposition_auto_only.json` | `Fig3/offset_vs_inherent_2_distributions.ipynb`; `Fig3/hsqc_offset_plots.ipynb` uses the first |

Figures 5 and S8–S11:

| Run first | It writes | Needed by |
|---|---|---|
| `Fig5/compute_md_justs2_justdihedral_persist90_panav_10seedens.ipynb` | `md_1us_combined_48_justs2_justdihedral_panav10seedens.json` | `Fig5/s2psi_ss_exposure_err_panav_10seedens.ipynb`, `Fig5/complexity_dynamics_relation.ipynb`, `Fig5/complexity_dynamics_relation_<model>.ipynb` |
| `Fig5/persist90_ss_burial_simple.ipynb` | `persist90_ss_burial_simple.csv` | `Fig5/s2psi_ss_exposure_err_panav_10seedens.ipynb`, `Fig5/get_persist_50.ipynb` |
| `Fig5/compute_md_dssp8_phipsi_gated_allframes.py` | `md_dssp8_phipsi_gated_allframes_panav10seedens.npz` | `Fig5/get_persist_50.ipynb` |
| `Fig5/get_persist_50.ipynb` | `md_phipsi_perframe_coil_persist50_panav10seedens.npz` | `Fig5/coil_rama_error_map.ipynb` |
| `SI/FigS8_S10_S16/be_sweep_bonferroni_panav_10seedens_<model>.ipynb` | `be_sweep_bonferroni_results_panav10seedens_<model>.csv` | S8e, last cell of `Fig5/be_sweep_bonferroni_panav_10seedens.ipynb` |

Several MD notebooks take 0.5–1.5 h each (`persist90_ss_burial_simple` ≈ 1.5 h, the `be_sweep_bonferroni` bootstraps ≈ 40–70 min); `compute_md_dssp8_phipsi_gated_allframes.py` takes about 30 min.
