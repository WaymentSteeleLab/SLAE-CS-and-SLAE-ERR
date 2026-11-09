# SLAE-CS and SLAE-ERR

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-yellow)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg?logo=python&logoColor=white)](https://python.org/downloads)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/WaymentSteeleLab/SLAE-CS/blob/main/notebooks/SLAE-CS_colab.ipynb)
[![Hugging Face model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-SLAE--CS-blue)](https://huggingface.co/hws-lab/SLAE-CS)
[![Hugging Face dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-rBMRB-blue)](https://huggingface.co/datasets/hws-lab/rBMRB)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.24023681-blue)](https://doi.org/10.5281/zenodo.24023681)

SLAE-CS and SLAE-ERR are the models of our paper "*Learning the limits of protein structure models with NMR chemical
shifts*".

Given a protein structure, SLAE-CS predicts the backbone amide ¹⁵N chemical shift of each residue, using the all-atom
SLAE encoder with a regression head. SLAE-ERR estimates the absolute error of each prediction from the encoder
embedding and the spread of the 10-seed SLAE-CS ensemble.

- Model weights on 🤗 Hugging Face: [hws-lab/SLAE-CS](https://huggingface.co/hws-lab/SLAE-CS) (10 SLAE-CS seeds and SLAE-ERR).
- Dataset on 🤗 Hugging Face: [hws-lab/rBMRB](https://huggingface.co/datasets/hws-lab/rBMRB), re-referenced BMRB chemical shifts
  (PANAV, LACS and no re-referencing) with train / val / test splits, AlphaFold2 structures and the SLAE-ERR training data.
- Data for the figures and the curation on [Zenodo](https://doi.org/10.5281/zenodo.24023681).
- A [Google Colab notebook](https://colab.research.google.com/github/WaymentSteeleLab/SLAE-CS/blob/main/notebooks/SLAE-CS_colab.ipynb)
  that runs SLAE-CS and SLAE-ERR without a local GPU.

In this repository we provide:
- [SLAE-CS and SLAE-ERR](#slae-cs-and-slae-err)
  - [Repository layout](#repository-layout)
  - [Installation](#installation)
  - [Quick start](#quick-start)
    - [Test the installation](#test-the-installation)
    - [Masked residues and SLAE-ERR](#masked-residues-and-slae-err)
  - [Training](#training)
  - [Figures](#figures)
  - [Baselines](#baselines)
  - [rBMRB curation](#rbmrb-curation)
  - [SLAE](#slae)
  - [Citation](#citation)
  - [License](#license)

## Repository layout

| path | contents |
|---|---|
| [`predict.py`](predict.py) | inference on new structures (SLAE-CS ensemble + SLAE-ERR) |
| [`examples/`](examples), [`tests/smoke_test.py`](tests/smoke_test.py) | three example structures with the reference output, and the test that compares them |
| [`notebooks/SLAE-CS_colab.ipynb`](notebooks/SLAE-CS_colab.ipynb) | install, test and prediction on a Google Colab GPU |
| [`slae_cs/`](slae_cs) | Training (`train/`), and inference on a structure set (`inference/`) |
| [`error_head/`](error_head) | SLAE-ERR training and inference, and the data helpers they use |
| [`configs/`](configs) | hydra configs of the training and inference scripts (defaults: the released settings) |
| [`scripts/`](scripts) | downloads (weights and data, analysis data, third-party code), flags file, evaluation, K-fold splits and SLAE-ERR table, ensemble merge, checkpoint conversion, NMR helpers, train-size subsets |
| [`docs/`](docs) | [reproduce.md](docs/reproduce.md): training, inference, evaluation; [figures.md](docs/figures.md): code of each figure panel |
| [`analysis/`](analysis/README.md) | figure notebooks and scripts, their data list and environment |
| [`curation/`](curation/README.md) | rBMRB and whole BMRB curation notebooks |
| [`baselines/`](baselines/README.md) | baselines of the paper (ESM3, MACE-OFF23, ProteinMPNN) |
| [`splits/`](splits) | id lists of the splits, the 5 K-fold folds and the train-size subsets |
| [`SLAE/`](SLAE) | SLAE encoder code (see [SLAE](#slae)) |
| [`containers/slae/`](containers/slae) | definition, environment and package list of the runtime image |

## Installation

The results of the paper were produced in an Apptainer image built from `containers/slae/slae.def`
(Python 3.11, PyTorch 2.6.0 + CUDA 12.4, PyTorch Geometric 2.7.0, PyTorch Lightning 1.9.5; the exact
package versions of that image are in `containers/slae/pip_list.txt`). Linux x86-64 with an NVIDIA
driver for CUDA 12.4 is assumed; `predict.py` also runs on the CPU.

**Apptainer** (building needs root; the image is about 9.1 GB):
```bash
cd containers/slae && sudo apptainer build slae.sif slae.def && cd ../..
apptainer exec --nv containers/slae/slae.sif bash -c \
  "source /opt/conda/etc/profile.d/conda.sh && conda activate slae && cd '$PWD' && python predict.py --help"
```
Run every command of this README the same way, inside `bash -c "..."`: the image's environment
changes into `/workspace`, so the explicit `cd` returns to the repository. If the repository is not
under your home directory, also bind it (`-B "$PWD"`).

**conda** (without Apptainer):
```bash
conda env create -f environment.yml && conda activate slae
pip install torch-scatter==2.1.2 torch-cluster==1.6.3 torch-sparse==0.6.18 torch-geometric==2.7.0 \
    -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

## Quick start

```bash
python scripts/download_from_hf.py                       # weights, data, structures, pretrained encoder
python predict.py --weights model/SLAE-CS-PANAV --error-model model/SLAE-ERR --pdb my_protein.pdb --out my_protein
```

- Download: `python scripts/download_from_hf.py --help` lists the options (`--scheme` for the LACS / BASELINE /
  whole-BMRB data, `--structures` to fetch only some splits, `--no-data` for the weights alone).
- Input: one protein per PDB file (a directory of PDB files also works). Only the first model of a
  multi-model file is used; split NMR ensembles with `scripts/inference_custom/split_nmr_models.py`.
- Output: `my_protein.csv`, one row per residue with `N_mean` and `N_std` (mean and spread of the
  10 seeds, ppm), `N_abs_error` (SLAE-ERR), and the prediction of each seed; `my_protein.json` with
  the same values per protein. Prolines are left out (no amide N).
- `--device auto|cuda|cpu` (default `auto`: the GPU if there is one); inference runs at batch size 1.

### Test the installation

In the SLAE-CS environment (see Installation), from the repository root:
```bash
python scripts/download_from_hf.py --no-data       # weights only
python tests/smoke_test.py                         # --quick: one protein only
```

`examples/structures/` holds three small proteins of the PANAV test set (AlphaFold2 models from
rBMRB, CC-BY-4.0) and `examples/expected/example_gpu.csv` the `predict.py` output for them in
the environment of the paper on an NVIDIA A100. The test runs `predict.py` on the three structures
and passes if every value agrees within 1e-3 ppm; other GPUs change the values by floating-point
rounding only. It takes under a minute on a GPU. `predict.py` and the test also run on the CPU
(`--device cpu`), more slowly.

Without a local GPU: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/WaymentSteeleLab/SLAE-CS/blob/main/notebooks/SLAE-CS_colab.ipynb)
`notebooks/SLAE-CS_colab.ipynb` installs SLAE-CS on a Google Colab GPU, runs this test and predicts an
uploaded structure.

### Masked residues and SLAE-ERR

In training and evaluation, two kinds of residues were masked: residues of affinity tags, protease
cleavage sites and their flanks (labels `T` / `t` in `tag_labels.csv`, assigned from the sequence),
and, on the validation and test sets, residues whose shift was flagged as unreliable during
re-referencing. For SLAE-CS this only removes them from the loss and the scores; its predictions
do not depend on the mask. For SLAE-ERR it also changes the input: its ensemble features (the
spread and the sorted, centred predictions of the 10 seeds) were set to 0 on masked residues.
`predict.py` has no tag labels or flags for a new structure and computes these features on every
residue. To reproduce the paper's SLAE-ERR predictions on the benchmark sets exactly, use
`error_head/infer_oof_abs_sorted.py` ([docs/reproduce.md, SLAE-ERR test split](docs/reproduce.md#slae-err-test-split-benchmark-proteins-only)).

Tags are labelled from the sequence by `curation/rBMRB/assign_labels.py`: `relabel_tags(sequence)` returns one
label per residue (`A` standard, `P` proline, `T` tag / cleavage motif, `t` flank).

## Training

[docs/reproduce.md](docs/reproduce.md) gives every command with the GPU time of each step:

1. [Setup](docs/reproduce.md#0-setup): data, structures and the pretrained SLAE encoder.
2. [SLAE-CS](docs/reproduce.md#1-slae-cs): the 10 full-data models and the train-size sweep.
3. [SLAE-ERR](docs/reproduce.md#2-slae-err): K-fold models, out-of-fold predictions, the training table, and SLAE-ERR itself.
4. [Inference](docs/reproduce.md#3-inference-with-the-released-weights) and
   [evaluation](docs/reproduce.md#4-evaluation) on the benchmark splits, with the numbers of the paper.
5. [Your own models with `predict.py`](docs/reproduce.md#5-using-your-own-models-with-predictpy): checkpoint conversion.

The defaults of the hydra configs in `configs/` are the settings of the released models.

## Figures

[analysis/](analysis/README.md): the notebooks of every figure panel. [docs/figures.md](docs/figures.md) lists the code of each
  panel and [analysis/DATA.md](analysis/DATA.md) the data. Data (Hugging Face and [Zenodo](https://doi.org/10.5281/zenodo.24023681)) and third-party code:
  `python scripts/download_analysis_data.py --parts figures` and `python scripts/setup_external.py`.

## Baselines

See [baselines/](baselines/README.md).

## rBMRB curation

[curation/](curation/README.md): the notebooks that built the rBMRB dataset from a BMRB dump, step by step. Inputs ([Zenodo](https://doi.org/10.5281/zenodo.24023681)):
  `python scripts/download_analysis_data.py --parts curation`.

## SLAE

`SLAE/` contains code from [SLAE](https://github.com/ProteinDesignLab/SLAE) (Chen et al.), distributed under
its licence (`LICENSE-SLAE`). The pretrained encoder weights (`autoencoder.ckpt`) are downloaded from the
SLAE repository by `scripts/download_from_hf.py`.

## Citation

Please cite the SLAE-CS paper (citation to be added on publication) and SLAE.

## License

Apache-2.0 (`LICENSE`). The files in `SLAE/` are distributed under the SLAE licence (`LICENSE-SLAE`).
