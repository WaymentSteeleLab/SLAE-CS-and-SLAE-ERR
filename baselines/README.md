# Baselines: ESM3, MACE-OFF23 and ProteinMPNN

Scripts of the encoder benchmark of the paper: other per-residue representations, with the same shift head, data,
masks and evaluation as SLAE-CS. Trained baseline weights and embeddings are not distributed. To predict shifts of new
structures, use SLAE-CS (`predict.py`).

| Baseline | Encoder | Input to the head | Head LR | RMSE (ppm) | MAE (ppm) |
|---|---|---|---|---|---|
| ESM3 seq+struct | ESM3-open, frozen | sequence + backbone embedding, 1536-d | 3e-4 | 1.8659 | 1.2776 |
| MACE-OFF23 | MACE-OFF23 large, frozen | invariant descriptors of atom N, all layers, 448-d | 3e-5 | 1.9233 | 1.3527 |
| ESM3 seq-only | ESM3-open, frozen | sequence-only embedding, 1536-d | 3e-4 | 2.1751 | 1.4984 |
| ProteinMPNN | `v_48_020` encoder, fine-tuned (encoder LR 1e-3) | encoder output, 128-d | 3e-5 | 2.2731 | 1.6123 |

Paper values: backbone amide ¹⁵N, 10-seed mean, rBMRB-PANAV test set, 407 proteins and 42,153 residues (SLAE-CS: 1.7447
and 1.1843).

## Third-party models

Get them from their sources, whose terms apply. The versions are those of the paper.

| Model | Source | Version | Downloaded by |
|---|---|---|---|
| ESM3-open | [EvolutionaryScale/esm3-sm-open-v1](https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1) | `47f0545` | `esm3/extract_esm3.py` |
| MACE-OFF23 large | [ACEsuit/mace-off](https://github.com/ACEsuit/mace-off) | `91a78c5` | `setup_third_party.py --mace --accept-asl`, or `mace/mace.def` |
| ProteinMPNN `v_48_020` | [dauparas/ProteinMPNN](https://github.com/dauparas/ProteinMPNN) | `8907e66` | `setup_third_party.py --proteinmpnn` |

## Setup

Follow step 0 of `docs/reproduce.md` (`R`, `PYTHONPATH`, `MASK`, `TAG`, data under `data/cs/`), then download what you need:

```bash
python baselines/setup_third_party.py --proteinmpnn
python baselines/setup_third_party.py --mace --accept-asl        # only outside the MACE image
```

Environments: SLAE-CS (`containers/slae/`) for training and inference; `esm3/esm3.def` and `mace/mace.def` for
extraction, with the package versions of the paper runs in `pip_list.txt` next to each. Training needs the GPUs given in
`docs/reproduce.md`; these environments do not run on NVIDIA Blackwell GPUs.

## Embeddings (ESM3 and MACE-OFF23)

```bash
cat train.list val.list test.list | sed 's/$/.pdb/' > all_pdbs.txt
mkdir -p emb_chunks

# in the ESM3 image
python baselines/esm3/extract_esm3.py --pdb_dir structures --list all_pdbs.txt --out_dir emb_chunks/esm3_all
tar czf emb_chunks/esm3_all.tar.gz -C emb_chunks esm3_all
python baselines/esm3/pack_embeddings.py --src emb_chunks --prefix esm3 --key seqstruct --key seqonly \
  --panav data/cs --list_dir . --out_dir emb

# in the MACE image
python baselines/mace/extract_mace.py --pdb_dir structures --list all_pdbs.txt --out_dir emb_chunks/mace_all \
  --model "$MACE_MODEL" --mode crop --rmax 5.0
tar czf emb_chunks/mace_all.tar.gz -C emb_chunks mace_all
python baselines/mace/pack_mace.py --src emb_chunks --panav data/cs --list_dir . --out_dir emb
```

## Training (seeds 1-10)

```bash
s=1; e=esm3_seqstruct; lr=3e-4       # esm3_seqonly: 3e-4; mace_N: 3e-5
python scripts/make_workdir.py runs/${e}_seed$s && cd runs/${e}_seed$s
python $R/baselines/embedding_head/embedding_shift_trainer.py hydra.run.dir=. run_name=${e}_seed$s \
  emb_npz=$R/emb/$e.npz emb_name=$e atom=N lr=$lr seed=$s mask_json=$MASK tag_mask_csv=$TAG
cd $R

s=1                                  # fine-tuned ProteinMPNN
python scripts/make_workdir.py runs/mpnn_ft_seed$s && cd runs/mpnn_ft_seed$s
python $R/baselines/proteinmpnn/mpnn_launcher.py hydra.run.dir=. run_name=mpnn_ft_seed$s atom=N seed=$s epochs=25 \
  lr=3e-5 lr_last=1e-3 ckpt=$PWD/unused.ckpt mask_json=$MASK tag_mask_csv=$TAG
cd $R
```

The model of each seed is the checkpoint with the best validation RMSE. Convert it for the predictors:

```bash
python scripts/to_safetensors.py runs/${e}_seed$s/best_ckpts/top3_val_N_epoch_RMSE_*.ckpt my_models/$e/seed$s
```

## Inference and evaluation

From a working directory made with `python scripts/make_workdir.py runs/test --structures splits/panav/test.list`:

```bash
python $R/baselines/embedding_head/emb_shift_predictor.py --ckpt $R/my_models/$e/seed$s --run_tag ${e}_seed$s \
  --emb_npz $R/emb/$e.npz --split test --data_dir . --mask_json $MASK --tag_mask_csv $TAG
python $R/baselines/proteinmpnn/mpnn_ft_predictor.py --ckpt $R/my_models/mpnn_ft/seed$s --run_tag mpnn_ft_seed$s \
  --split test --mask_json $MASK --tag_mask_csv $TAG
python $R/scripts/evaluate.py cs --gt $R/data/cs/test.json --pred predictions_N_${e}_seed*_test.json
```

## References

ESM3: Hayes et al., *Science* (2025). MACE-OFF23: Kovács et al., *J. Am. Chem. Soc.* (2025). Local protein
environments with MACE: Bojan et al., arXiv:2505.23354 (code: https://github.com/mb012/MLFF_representation).
ProteinMPNN: Dauparas et al., *Science* (2022).
