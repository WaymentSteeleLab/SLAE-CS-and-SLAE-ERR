# Reproducing the paper: training → inference → evaluation

All commands run in the SLAE-CS environment ([Installation](../README.md#installation)) on one GPU. The hydra
configs in `configs/` hold the settings of the released models as defaults, so the commands only give paths,
seeds and run names (`key=value`).

## 0. Setup

```bash
git clone https://github.com/WaymentSteeleLab/SLAE-CS.git SLAE-CS && cd SLAE-CS
python scripts/download_from_hf.py                               # weights, PANAV data, structures, autoencoder.ckpt
export R=$PWD PYTHONPATH=$PWD WANDB_MODE=offline
export CKPT0=$R/autoencoder.ckpt TAG=$R/data/cs/tag_labels.csv \
       MASK=$R/data/cs/bmrb_cs_data_withoffset_panav_flags_only.json \
       OOF=$R/data/cs/train_val_test_oof_ensemble_10seed_N.json
```
For LACS / BASELINE / whole-BMRB models, download with `--scheme LACS|BASELINE|WHOLE` into another `--dest`. Every
run gets its own working directory (`scripts/make_workdir.py`), because the scripts write their checkpoints and
predictions where they are started.

Resources, measured on the cluster runs of the paper (30 GB RAM requested). Training ran on GPUs with 80 GB or
more: SLAE-CS peaked at 60-82 GB, SLAE-ERR at up to 109 GB. A 40 GB GPU runs out of memory on batches of large
proteins at batch size 2. Inference on the test set used up to about 39 GB.

| step | runs | GPU time |
|---|---|---|
| SLAE-CS, 25 epochs, PANAV / LACS / BASELINE data | 10 seeds per scheme | ≈ 5 h each |
| SLAE-CS, whole-BMRB training set | 10 seeds | ≈ 11 h each |
| K-fold SLAE-CS models (out-of-fold predictions) | 5 folds × 10 seeds = 50 | ≈ 4.5 h each, ≈ 220 GPU-h in total |
| SLAE-ERR, 10 epochs | 1 | 1–2 h |
| train-size sweep | 9 fractions × 4 schemes | ≤ 5 h each |
| inference, PANAV test set (413 proteins) | per seed | ≈ 2 min |

The K-fold step in particular needs a GPU cluster. The first run on a structure set also featurises it into
`processed_structures/` (cache).

## 1. SLAE-CS

### Full-data models (10 seeds)
```bash
for s in 1 2 3 4 5 6 7 8 9 10; do
  python scripts/make_workdir.py runs/panav_seed$s && cd runs/panav_seed$s
  python $R/slae_cs/train/chemical_shift_trainer.py hydra.run.dir=. run_name=panav_seed$s \
    ckpt=$CKPT0 mask_json=$MASK tag_mask_csv=$TAG seed=$s
  cd $R
done
```
The model of each seed is the checkpoint with the best validation RMSE,
`best_ckpts/top3_val_N_epoch_RMSE_epoch<E>_<rmse>.ckpt`. The same command trains the LACS, BASELINE and whole-BMRB
models from their data roots.

### Train-size sweep (seed 1)
One model per fraction f = 10, 20, …, 90, using the nested subsets in `splits/train_size_sweep/`:
```bash
python scripts/make_workdir.py runs/panav_trainfrac$f && cd runs/panav_trainfrac$f
python $R/slae_cs/train/chemical_shift_trainer.py hydra.run.dir=. run_name=panav_trainfrac$f \
  ckpt=$CKPT0 mask_json=$MASK tag_mask_csv=$TAG seed=1 \
  train_list=$R/splits/train_size_sweep/panav_lacs_noref/train_$f.list
cd $R
```
For whole-BMRB, use `noref_whole/` instead of `panav_lacs_noref/`. The 100 % point is the full-data seed-1 model.

## 2. SLAE-ERR

Step 0 downloads the training table, so 2a–2c are only needed to rebuild it. To train SLAE-ERR on the released
table, go to the second command of 2c.

### 2a. K-fold models and out-of-fold predictions
The folds are in `splits/kfold/`; `scripts/kfold/make_kfold_splits.py` rebuilds them. For each fold f = 0..4 and
seed s = 1..10:
```bash
python scripts/make_workdir.py runs/kfold${f}_seed$s && cd runs/kfold${f}_seed$s
python $R/slae_cs/train/chemical_shift_trainer.py hydra.run.dir=. run_name=kfold${f}_seed$s \
  ckpt=$CKPT0 mask_json=$MASK tag_mask_csv=$TAG seed=$s \
  train_list=$R/splits/kfold/train_fold$f.list test_list=$R/splits/kfold/heldout_fold$f.list \
  test_json=$R/data/cs/train.json +flags_on_test=false
cd $R
```
The held-out fold is the "test" split of these runs; it gets the tag mask only. Then the
out-of-fold predictions of each (fold, seed) model on its held-out fold, in a working directory with only
that fold's structures:
```bash
python scripts/make_workdir.py oof/fold$f --structures splits/kfold/heldout_fold$f.list && cd oof/fold$f
python $R/slae_cs/inference/chemical_shift_predictor.py hydra.run.dir=. \
  ckpt=$(ls $R/runs/kfold${f}_seed$s/best_ckpts/top3_val_N_epoch_RMSE_*.ckpt) \
  atom=N method=panav_kfold${f}_seed$s tag_mask_csv=$TAG
cd $R
```

### 2b. Validation and test predictions of the 10 full-data models
For split = val and test, and seed s = 1..10:
```bash
python scripts/make_workdir.py preds/$split --structures splits/panav/$split.list && cd preds/$split
python $R/slae_cs/inference/chemical_shift_predictor.py hydra.run.dir=. \
  ckpt=$(ls $R/runs/panav_seed$s/best_ckpts/top3_val_N_epoch_RMSE_*.ckpt) \
  atom=N method=panav_seed${s}_full_$split mask_json=$MASK tag_mask_csv=$TAG
cd $R
```

### 2c. Training table and SLAE-ERR training
```bash
python scripts/kfold/collect_runs.py --runs runs --oof oof --out lists   # checkpoint lists, oof/all/
python scripts/kfold/merge_oof.py --data_dir data/cs --folds_dir splits/kfold \
  --kfold_pred_dir oof/all --kfold_ckpt_list lists/kfold_ckpts.csv \
  --val_pred_dir preds/val --val_ckpt_list lists/full_ckpts.csv \
  --test_pred_dir preds/test --test_ckpt_list lists/full_ckpts.csv --out $OOF
```
SLAE-ERR is then trained on the table:
```bash
python scripts/make_workdir.py runs/slae_err && cd runs/slae_err
python $R/error_head/chemical_shift_trainer_justerr_oof_sorted.py hydra.run.dir=. run_name=slae_err \
  ckpt=$CKPT0 mask_json=$MASK tag_mask_csv=$TAG oof_json=$OOF
cd $R
```
The released model is the checkpoint with the best validation Spearman(mu_err, |residual|).

## 3. Inference with the released weights

### SLAE-CS, PANAV test set (10 seeds)
```bash
python scripts/make_workdir.py runs/test --structures splits/panav/test.list && cd runs/test
for s in 01 02 03 04 05 06 07 08 09 10; do
  python $R/slae_cs/inference/chemical_shift_predictor.py hydra.run.dir=. \
    ckpt=$R/model/SLAE-CS-PANAV/seed$s/model.safetensors atom=N method=seed${s}_test \
    mask_json=$MASK tag_mask_csv=$TAG
done
python $R/scripts/ensemble/merge_seed_predictions.py --pred predictions_N_seed{01..10}_test.json \
  --seeds 1 2 3 4 5 6 7 8 9 10 --out predictions_N_ensemble_test.json
cd $R
```

### SLAE-ERR, test split (benchmark proteins only)
For the proteins of the SLAE-ERR training table only: the input features are read from the table, so other
proteins are skipped. Needs the train / val / test structures (the data module builds all three splits):
```bash
python error_head/infer_oof_abs_sorted.py hydra.run.dir=. ckpt=$CKPT0 \
  +trained_ckpt=$R/model/SLAE-ERR/model.safetensors mask_json=$MASK tag_mask_csv=$TAG oof_json=$OOF \
  method=slae_err
```
This reproduces the paper's SLAE-ERR predictions exactly; for the difference from `predict.py`, see
[Masked residues and SLAE-ERR](../README.md#masked-residues-and-slae-err-todo).

### New structures
See [Quick start](../README.md#quick-start). For NMR ensembles, use the helpers in `scripts/inference_custom/`:
`split_nmr_models.py --src <nmr pdbs> --out <dir>` and `check_inputs.py --dir <dir>`, then `predict.py --pdb <dir>`, then
`aggregate_nmr_ensemble.py --ensemble <prefix>.json --manifest <dir>/nmr_models.tsv --structures <dir> --out <prefix>_nmr`.

## 4. Evaluation

```bash
python scripts/evaluate.py cs --gt data/cs/test.json --pred runs/test/predictions_N_seed*_test.json
#   paper: 407 proteins, 42153 residues, RMSE 1.7447 (10-seed ensemble)
python scripts/evaluate.py error --pred predictions_N_slae_err_err.json
#   paper: 42153 residues, MAE |mu_err - |residual|| 0.760, Spearman 0.3207
```
Scoring protocol: see the docstring of `scripts/evaluate.py`.

## 5. Using your own models with predict.py

The trainers save Lightning checkpoints; `predict.py` reads `model.safetensors`. Convert the best
checkpoint of each run into the layout of the released weights:
```bash
for s in 1 2 3 4 5 6 7 8 9 10; do
  python scripts/to_safetensors.py runs/panav_seed$s/best_ckpts/top3_val_N_epoch_RMSE_*.ckpt \
    my_models/SLAE-CS-PANAV/seed$(printf %02d $s)
done
python scripts/to_safetensors.py runs/slae_err/best_ckpts/top3_val_N_epoch_mu_err_spearman_*.ckpt my_models/SLAE-ERR
python predict.py --weights my_models/SLAE-CS-PANAV --error-model my_models/SLAE-ERR --pdb <file or dir> --out <prefix>
```
