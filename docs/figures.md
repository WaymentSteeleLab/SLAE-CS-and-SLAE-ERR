# Code of each figure panel

Paths are relative to the repository root. Run each notebook from its own folder; `analysis/README.md` gives the order where one notebook needs the output of another, and `analysis/DATA.md` the data.

| Panel | Code | Note |
|---|---|---|
| 1b,1c | `analysis/Fig1/make_hsqc_example.ipynb` |  |
| 1d | `analysis/Fig1/panav_N_distribution.ipynb` |  |
| 1e | `analysis/Fig1/bmrb_source_comparison.ipynb` |  |
| 2b | `analysis/Fig2/offset_violin_styled.ipynb` |  |
| 2c | `analysis/Fig2/rmse_comparison_results_10seedens.ipynb` |  |
| 2e | `analysis/Fig2/kendalltau_comparison_results_10seedens.ipynb` |  |
| 2f | `analysis/Fig2/correlation_ranking_offset.ipynb` |  |
| 2g | `analysis/Fig2/eval_kendall_saturation.ipynb` |  |
| 3b, 3c (input); Table S2/S3 | `analysis/Fig3/binding_vs_corr_rmse.ipynb` | run after offset_vs_inherent_1_decomposition; writes decomposition_no_othernoise.json, decomposition_auto_only.json |
| 3b, 3c; S7 | `analysis/Fig3/offset_vs_inherent_2_distributions.ipynb` | run after binding_vs_corr_rmse |
| 3d | `analysis/Fig3/hsqc_offset_plots.ipynb` | run after binding_vs_corr_rmse |
| 4b,c | `analysis/Fig4/error_corr_vs_slaecs_panav10seedens.ipynb` |  |
| 4d | `analysis/Fig4/residue_ss_sasa_error_median_panav_10seedens.ipynb` |  |
| 5a | `analysis/Fig5/ss_effect_sweep_panav_10seedens.ipynb` |  |
| 5b | `analysis/Fig5/md_protein_run48_profile.ipynb` | selection of the 48 MD proteins |
| 5c, 5d, 5f; S9, S10, S11 (input) | `analysis/Fig5/persist90_ss_burial_simple.ipynb` | writes persist90_ss_burial_simple.csv |
| 5c, 5g, 5h; S9, S10 (input) | `analysis/Fig5/compute_md_justs2_justdihedral_persist90_panav_10seedens.ipynb` | writes md_1us_combined_48_justs2_justdihedral_panav10seedens.json |
| 5c; S9; S10a, S10c | `analysis/Fig5/s2psi_ss_exposure_err_panav_10seedens.ipynb` | S9 uses the Table S1 values: external/makeshift/makeshift/data/bmrb_stats.csv |
| 5d, 5f, S11 (helper) | `analysis/Fig5/hk_rama_regions.py` | H&K 2010 Ramachandran regions from the digitized Fig 6a boundaries |
| 5d, 5f, S11 (input) | `analysis/Fig5/get_persist_50.ipynb` | writes md_phipsi_perframe_coil_persist50_panav10seedens.npz |
| 5d, 5f, S11 (input) | `analysis/Fig5/compute_md_dssp8_phipsi_gated_allframes.py` | per-frame DSSP and phi/psi of the gated residues; input of get_persist_50 |
| 5d, 5f; S11 | `analysis/Fig5/coil_rama_error_map.ipynb` | imports hk_rama_regions.py; writes the H&K label cache |
| 5e; S8e | `analysis/Fig5/be_sweep_bonferroni_panav_10seedens.ipynb` | the last cell draws S8e; run the four S8e input notebooks first |
| 5g | `analysis/Fig5/complexity_dynamics_relation.ipynb` |  |
| 5g | `analysis/Fig5/complexity_dynamics_relation_esm3.ipynb` | same matrix for this model |
| 5g | `analysis/Fig5/complexity_dynamics_relation_esm3_seqonly.ipynb` | same matrix for this model |
| 5g | `analysis/Fig5/complexity_dynamics_relation_mace.ipynb` | same matrix for this model |
| 5g | `analysis/Fig5/complexity_dynamics_relation_mpnn_ft.ipynb` | same matrix for this model |
| 5h | `analysis/Fig5/enrichment_structures/generate_pml_pr10.ipynb` |  |
| 5h | `analysis/Fig5/enrichment_structures/generate_pml_19743_hotcells.ipynb` |  |
| 5h | `analysis/Fig5/enrichment_structures/draw_colorbar.ipynb` | color scale |
| 5i | `analysis/Fig5/case_studies/write_case_pml_3cases.ipynb` |  |
| 6 (value) | `analysis/Fig6/eval_oracle_mae.ipynb` | lower bound of the SLAE-ERR MAE for 0.5 ppm experimental noise |
| 6b | `analysis/Fig6/eval_error_auroc.ipynb` |  |
| 6c; S15; S16 | `analysis/Fig6/Grouping_geom_allmodels.ipynb` |  |
| 6e | `analysis/Fig6/analyze_oof_err_predictions_sorted_centered.ipynb` |  |
| 6f | `analysis/Fig6/eval_error_threshold.ipynb` |  |
| 6g | `analysis/Fig6/structure_error_figs/write_gt_error_bfactor_ubiquitin.ipynb` |  |
| 6g, 6h | `analysis/Fig6/structure_error_figs/draw_colorbar.ipynb` | color scale |
| 6h | `analysis/Fig6/structure_error_figs/write_error_bfactor_pdbs.ipynb` |  |
| S1 | `analysis/SI/FigS1/sample_random_af2_structures.ipynb` |  |
| S1 | `analysis/SI/FigS1/ucbshift_hsqc_offset_plots.ipynb` |  |
| S2 | `analysis/SI/FigS2/distribution_analysis_acrossmodel.ipynb` |  |
| S4 (input) | `analysis/SI/FigS4/merge_seed_predictions.py` |  |
| S4a | `analysis/SI/FigS4/analyze_seed_stability_10_kendall.ipynb` |  |
| S4b | `analysis/SI/FigS4/analyze_mcdropout_stability_10_kendall.ipynb` |  |
| S5 | `analysis/SI/FigS5/error_distribution_panav_10seedens.ipynb` |  |
| S6a, S6b; 3b, 3c (input) | `analysis/Fig3/offset_vs_inherent_1_decomposition.ipynb` | writes decomposition.json; run before binding_vs_corr_rmse |
| S8a | `analysis/SI/FigS8_S10_S16/ss_effect_sweep_panav_10seedens_esm3.ipynb` |  |
| S8b | `analysis/SI/FigS8_S10_S16/ss_effect_sweep_panav_10seedens_mace.ipynb` |  |
| S8c | `analysis/SI/FigS8_S10_S16/ss_effect_sweep_panav_10seedens_esm3_seqonly.ipynb` |  |
| S8d | `analysis/SI/FigS8_S10_S16/ss_effect_sweep_panav_10seedens_mpnn_ft.ipynb` |  |
| S8e (input) | `analysis/SI/FigS8_S10_S16/be_sweep_bonferroni_panav_10seedens_esm3.ipynb` | writes be_sweep_bonferroni_results_panav10seedens_esm3.csv |
| S8e (input) | `analysis/SI/FigS8_S10_S16/be_sweep_bonferroni_panav_10seedens_esm3_seqonly.ipynb` | writes be_sweep_bonferroni_results_panav10seedens_esm3_seqonly.csv |
| S8e (input) | `analysis/SI/FigS8_S10_S16/be_sweep_bonferroni_panav_10seedens_mace.ipynb` | writes be_sweep_bonferroni_results_panav10seedens_mace.csv |
| S8e (input) | `analysis/SI/FigS8_S10_S16/be_sweep_bonferroni_panav_10seedens_mpnn_ft.ipynb` | writes be_sweep_bonferroni_results_panav10seedens_mpnn_ft.csv |
| S10b | `analysis/SI/FigS8_S10_S16/nheavy_neighbor_ss_exposure_panav_10seedens.ipynb` |  |
| S12 | `analysis/SI/FigS12/helix_capping_error_panav10seedens.ipynb` |  |
| S14a | `analysis/SI/FigS14/pack_N8_error_corr.ipynb` |  |
| S14b | `analysis/SI/FigS14/kendalltau_s2_4pub.ipynb` |  |
| S14c | `analysis/SI/FigS14/kendalltau_pex_4pub.ipynb` |  |
| S17 | `analysis/SI/FigS17/write_gt_error_bfactor_27512.ipynb` |  |
| S18 | `analysis/SI/FigS18/sequence_alignment.ipynb` |  |
| S19; S20 | `analysis/SI/FigS19_S20/seq_tm_err.ipynb` |  |
| S21a, S21b | `analysis/SI/FigS21/seq_tm_err_esm3_seqonly.ipynb` |  |
| S22 | `analysis/SI/FigS22/11547_af2template/` | write_gt_error_bfactor_11547_1_af2nmr.ipynb, its two AF2 + template PDBs, and analysis_11547_1/ |
| Per-residue features | `analysis/shared/feature_pipeline/merge_features.py` | input of Fig 4d, 5, 6 and SI; order in analysis/README.md |
| Per-residue features | `analysis/shared/feature_pipeline/add_plddt_ph_temp.py` | input of Fig 4d, 5, 6 and SI; order in analysis/README.md |
| Per-residue features | `analysis/shared/feature_pipeline/compute_geom_features.py` | input of Fig 4d, 5, 6 and SI; order in analysis/README.md |
| Per-residue features | `analysis/shared/feature_pipeline/dssp_calculation_test.ipynb` | input of Fig 4d, 5, 6 and SI; order in analysis/README.md |
| Per-residue features | `analysis/shared/feature_pipeline/sasa_exposure_test.ipynb` | input of Fig 4d, 5, 6 and SI; order in analysis/README.md |
| Per-residue features | `analysis/shared/feature_pipeline/extract_foldseek_3di_af2.ipynb` | input of Fig 4d, 5, 6 and SI; order in analysis/README.md |
| Per-residue features | `analysis/shared/feature_pipeline/extract_3dn_per_residue.ipynb` | input of Fig 4d, 5, 6 and SI; order in analysis/README.md |
| Per-residue features | `analysis/shared/feature_pipeline/run_talosn_batch.ipynb` | input of Fig 4d, 5, 6 and SI; order in analysis/README.md |
| text (MAE) | `analysis/Fig6/error_model_mae_summary.ipynb` | MAE of SLAE-ERROR quoted in text |
