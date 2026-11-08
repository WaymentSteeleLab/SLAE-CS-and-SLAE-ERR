# Baselines: ESM3, MACE-OFF23 and ProteinMPNN

The foundation-model baselines of the paper: other per-residue representations, with the same prediction trunk, data,
masks and evaluation as SLAE-CS.

| Baseline | Encoder | Input to the head | Head LR | RMSE (ppm) | MAE (ppm) |
|---|---|---|---|---|---|
| ESM3 seq+struct | ESM3-open, frozen | sequence + backbone embedding, 1536-d | 3e-4 | 1.8659 | 1.2776 |
| MACE-OFF23 | MACE-OFF23 large, frozen | invariant descriptors of atom N, both interaction layers, 448-d | 3e-5 | 1.9233 | 1.3527 |
| ESM3 seq-only | ESM3-open, frozen | sequence-only embedding, 1536-d | 3e-4 | 2.1751 | 1.4984 |
| ProteinMPNN | `v_48_020` encoder, fine-tuned (encoder LR 1e-3) | encoder output, 128-d | 3e-5 | 2.2731 | 1.6123 |

Paper values: backbone amide ¹⁵N, 10-seed ensemble (mean of the 10 seed predictions), rBMRB-PANAV test set,
407 proteins and 42,153 residues (SLAE-CS: 1.7447 and 1.1843).

## Third-party models

Get them from their sources, whose terms apply. The versions are those of the paper.

| Model | Source | Version |
|---|---|---|
| ESM3-open | [EvolutionaryScale/esm3-sm-open-v1](https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1) | `47f0545` |
| MACE-OFF23 large | [ACEsuit/mace-off](https://github.com/ACEsuit/mace-off) | `91a78c5` |
| ProteinMPNN `v_48_020` | [dauparas/ProteinMPNN](https://github.com/dauparas/ProteinMPNN) | `8907e66` |

## References

ESM3: Hayes et al., *Science* (2025). MACE-OFF23: Kovács et al., *J. Am. Chem. Soc.* (2025). Local protein
environments with MACE: Bojan et al., arXiv:2505.23354 (code: https://github.com/mb012/MLFF_representation).
ProteinMPNN: Dauparas et al., *Science* (2022).
