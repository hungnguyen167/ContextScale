# ContextScale - A Sentiment-Based Approach to Measuring Multidimensional Party Positions using Transformer

Replication repository for:

> Nguyen, Hung H. V. "A Sentiment-Based Approach to Measuring Multidimensional Party Positions using Transformer." *Political Science Research and Methods* (Forthcoming, 2026). 

**ContextScale** is a transformer-based framework for scaling party positions from political texts. It uses a deep ensemble of XLM-RoBERTa models with a shared attention architecture (`ContextScalePrediction`) to simultaneously predict topic categories and sentiment/stance from quasi-sentence level manifesto data, producing uncertainty-quantified party position scores. The approach is validated against Wordfish and Doc2Vec (Rheault & Cochrane 2020), and is extended to coalition agreements and Twitter data.

Final datasets and trained model weights are shared via [PSRM Dataverse](https://dataverse.harvard.edu/dataverse/PSRM).

---

## Repository Structure

```
ContextScale/
├── 01_manifesto_pull.Rmd      # Step 1 (R):      Pull manifesto data from CMP API
├── 02_r_prep.Rmd              # Step 2 (R):      Coalition agreement prep
├── 03_main_analyses.ipynb     # (Legacy combined notebook — superseded by 03a + 03b below)
├── 03a_main_analyses.ipynb    # Step 3a (Python): Main training, validity checks, scaling
├── 03b_ablation_study.ipynb   # Step 3b (Python): Ablation study only (optional, long)
├── 04_r_prep.Rmd              # Step 4 (R):      Wordfish benchmarks (runs AFTER Step 3a)
├── 05_visualizations.Rmd      # Step 5 (R):      All figures and tables in the paper
├── train.py                   # Standalone training script (CLI interface)
├── renv.lock                  # Pinned R package versions (restored via renv::restore())
├── scripts/
│   ├── replicate.sh           # End-to-end pipeline runner (env setup + all steps)
│   └── clean.sh               # Remove all generated outputs (keeps raw data + models)
├── utils/
│   ├── functions.py           # Data preprocessing, training/eval loops, scaling
│   ├── models.py              # ContextScalePrediction model definition
│   └── uncertainty.py         # Deep ensemble training and inference
├── data/
│   ├── CMP/                   # Comparative Manifesto Project metadata
│   ├── MOTN/                  # Twitter ground-truth datasets
│   ├── r_outputs/             # Outputs from R scripts (manifesto CSVs, Wordfish)
│   ├── py_outputs/            # Outputs from Python scripts (preprocessed manifesto, Doc2Vec, CS scores)
│   └── coalitionagree/        # Hand-coded coalition agreement texts
└── results/
    ├── classification results/ # F1/precision/recall tables per model variant
    ├── datasets/              # Final scaled datasets
    ├── models/                # Saved model weights (.safetensors)
    └── tabs and figs/         # Figures and tables for the paper
```

---

## Quick Start

Two convenience scripts in `scripts/` cover the most common workflows.

### Clean the workspace (prepare for a fresh run)
```bash
# Preview what will be deleted (dry-run, safe to run):
bash scripts/clean.sh

# Actually delete all generated outputs (keeps raw data + model weights):
bash scripts/clean.sh --confirm
```

### Run the full pipeline
```bash
# Conda environment (default) — installs env + runs all steps:
bash scripts/replicate.sh

# Use pip + virtualenv instead of Conda:
bash scripts/replicate.sh --venv

# Skip Python environment creation (already activated):
bash scripts/replicate.sh --skip-env

# Skip ablation notebook (faster run):
bash scripts/replicate.sh --skip-ablation

# Skip R package installation (already installed):
bash scripts/replicate.sh --skip-r-install

# Run in a detachable tmux session (recommended for long/remote runs):
bash scripts/replicate.sh --tmux

# Combine flags — e.g. run in tmux without reinstalling anything:
bash scripts/replicate.sh --tmux --skip-env --skip-r-install --skip-ablation
```

A timestamped log is written to `logs/replicate_YYYYMMDD_HHMMSS.log`.

**tmux quick-reference** (while attached):

| Action | Keys |
|--------|------|
| Detach (leave running) | `Ctrl-B` then `D` |
| Attach to session | `tmux attach-session -t contextscale-rep` |
| Follow log without attaching | `tail -f logs/replicate_*.log` |
| Kill session | `tmux kill-session -t contextscale-rep` |

---

## Replication Instructions

The pipeline runs in this order: **1 → 2 → 3a → 3b → 4 → 5**.
Steps 1, 2, 4, and 5 run in **R**; Steps 3a and 3b run in **Python**.

> `04_r_prep.Rmd` (Wordfish models) must run **after** `03a_main_analyses.ipynb` because
> the Wordfish scripts require the preprocessed manifesto data (`data/py_outputs/manifesto.csv`)
> and processed Twitter data (`data/py_outputs/tw_processed.csv`) that the Python notebook produces.

### Prerequisites

**R packages** — restore the exact pinned environment using [`renv`](https://rstudio.github.io/renv/):
```r
install.packages("renv")
renv::restore()
```

Or install manually:
```r
install.packages("pacman")
pacman::p_load(tidyverse, manifestoR, quanteda, quanteda.textmodels,
               ggplot2, viridis, ggsci, ggpubr, ragg, scales, gt,
               rstatix, gridExtra, openxlsx, countrycode, scico, caret,
               ggExtra, rempsyc, flextable, magrittr, grid, readr, dplyr)
```

**Python environment (recommended — Conda):**
```bash
conda env create -f environment.yml
conda activate contextscale
```

Or via pip:
```bash
pip install -r requirements.txt
```

See [`requirements.txt`](requirements.txt) and [`environment.yml`](environment.yml) for pinned package versions (Python 3.11.14, PyTorch 2.9.0).

A CUDA-capable GPU (≥ 16 VRAM) is strongly recommended for Steps 3a and 3b. Replicators may change the batch size if there is Out-Of-Memory error. Changing batch size is not expected to change inference results, but may very slightly change training outcomes (negligible).

**Manifesto API key:** Register for a free key at [manifesto-project.wzb.eu](https://manifesto-project.wzb.eu/). Create a file `manifesto_apikey.txt` in the repository root and paste your key into it:
```bash
echo "YOUR_API_KEY_HERE" > manifesto_apikey.txt
```
This file is git-ignored and will never be committed. Alternatively, download the pre-generated manifesto CSVs directly from [PSRM Dataverse](https://doi.org/10.7910/DVN/AFBN8X) and skip Step 1.

---

### Step 1 — Pull Manifesto Data (`01_manifesto_pull.Rmd`)

This R script uses the `manifestoR` package to download annotated manifesto quasi-sentences from the Comparative Manifesto Project (CMP) API.

- **Training countries** (17 Western European democracies): Austria, Belgium, Denmark, Finland, France, Germany, Greece, Iceland, Ireland, Italy, Netherlands, Norway, Portugal, Spain, Sweden, Switzerland, United Kingdom
- **Test countries** (out-of-sample, unseen languages): Czech Republic, Japan, Turkey, Russia, South Korea

**Outputs:**
- `data/r_outputs/pulled_manifestoes.csv` — training corpus
- `data/r_outputs/pulled_manifestoes_test.csv` — out-of-sample test corpus

---

### Step 2 — R Preparation (`02_r_prep.Rmd`)

Runs immediately after Step 1. Produces inputs that `03a_main_analyses.ipynb` and `03b_ablation_study.ipynb` need.

**What it does:**
1. **Coalition agreements** — reads the hand-coded coalition agreement Excel files and saves `data/r_outputs/coalitionagree_texts.csv` (used by Section 3.6 of `03a_main_analyses.ipynb`).

---


### Step 3a — Main Analyses (`03a_main_analyses.ipynb`)

This is the core Python notebook. Run cells sequentially. A GPU is required for training; all other cells (loading, inference, scaling) can run on CPU but will be much slower.

The notebook is divided into the following sections:

#### 3.1 Data Preparation

- Loads `data/r_outputs/pulled_manifestoes.csv`
- Applies `sentiment_code` and `topic_code` to assign labels from `cmp_code`
- Groups short quasi-sentences into longer sequences using `group_texts` (max 5x grouping factor, to fit within the 512-token limit)
- Saves preprocessed data to `data/py_outputs/manifesto.csv` and `data/py_outputs/manifesto_regrouped.csv`
- Tokenizes data and builds stratified train/eval/test splits

This step is done automatically in `train.py` if `data/py_outputs/manifesto.csv` and `data/py_outputs/manifesto_regrouped.csv` did not already exist. 

#### 3.2 Deep Ensemble Training & Inference
The **main ensemble** (`results/models/ensemble/`) and the **scaling ensemble** (`results/models/ensemble_scaling/`) are trained via `train.py` (see below). The notebook then:
- Loads those weights and runs inference on the full manifesto corpus
- Trains a small **10%-data unseen ensemble** on languages unseen during training (`results/models/manifesto_ensemble_dl_10/`) initialized from `ensemble_scaling/` weights — this training cell runs inside the notebook
- Saves results to `results/datasets/ensemble_results_test.pkl` (test-split predictions), `results/datasets/ensemble_test_dataset.csv` (test-split merged CSV), `results/datasets/ensemble_results_full.pkl` (full-corpus predictions), and `results/datasets/ensemble_full_dataset.csv` (full-corpus merged CSV)

#### 3.3 Validity Checks — Out-of-Sample Languages
- Tests the trained ensemble on the out-of-sample countries (Czech Republic, Japan, Turkey, Russia, South Korea) to assess cross-lingual transfer capability
- Evaluates both zero-shot transfer (no fine-tuning) and fine-tuned variants (10%-data ablation)
- Computes classification metrics (F1, precision, recall) per language and per architecture variant
- Results saved to `results/classification results/` with cross-lingual transfer analysis 
  
#### 3.4 Validity Checks — Coalition Agreements
- Tests the trained ensemble on hand-coded coalition agreement quasi-sentences from `data/r_outputs/coalitionagree_texts.csv`
- Evaluates both zero-shot transfer and fine-tuned variants (10%-data ablation)
- Computes classification metrics (F1, precision, recall) on out-of-domain political text
- Results saved to `results/classification results/` with domain-transfer analysis
  
#### 3.5 Validity Checks - Transfer to Twitter Data
- Loads three ground-truth Twitter datasets from `data/MOTN/` (Trump stance, Kavanaugh, Women's March)
- Recodes stance labels to left/right using `recode_tw`
- Fine-tunes a Twitter ensemble (`results/models/tw_ensemble/`, 5 models × 5 epochs) on the Twitter data — **this training cell runs inside the notebook**
- Saves results to `data/py_outputs/cs_tw.csv`

#### 3.6 Doc2Vec Scaling (Rheault & Cochrane 2020 replication)
- Trains country-level Doc2Vec models (vector size 500, PCA to 2D) as a baseline comparison
- General, topic-specific, environment-protection, welfare, and Twitter variants
- Outputs saved to `data/py_outputs/r&c_*.csv`


---

### Step 3b — Ablation Study (`03b_ablation_study.ipynb`)

This notebook runs only the architecture ablation and saves model weights + classification tables.

| Variant | Description | Save path |
|---|---|---|
| `base` | Standard XLM-RoBERTa + two independent heads | `results/models/manifesto_ContextScalePrediction_base/` |
| `sf` | Simple (unidirectional) information flow between heads | `results/models/manifesto_ContextScalePrediction_sf/` |
| `sa` | **Shared attention** (proposed architecture) | `results/models/manifesto_ContextScalePrediction_sa/` |
| `dg` | Dynamic gating between heads | `results/models/manifesto_ContextScalePrediction_dg/` |

Classification metrics (F1, precision, recall, accuracy per topic/sentiment class) are saved to `results/classification results/`.

---


### Standalone Training (`train.py`)

`train.py` is a standalone CLI script for training the two manifesto ensembles that the notebook expects to find before running. Run these **before** opening `03a_main_analyses.ipynb`.

#### To replicate `results/models/ensemble/` (train/eval split)

This ensemble is used for evaluation and validity checks in the notebook.

```bash
# Start a persistent tmux session first:
tmux new-session -d -s training
tmux attach-session -t training
conda activate contextscale

# Train (matches the repo config exactly):
python train.py
# Detach with Ctrl+B, D  |  Re-attach: tmux attach-session -t training
```

#### To replicate `results/models/ensemble_scaling/` (full dataset, no eval)

This ensemble is trained on the **entire** manifesto corpus (no held-out eval set) and is used for position scaling throughout the paper. It must be trained **before** running the notebook, as the 10%-data ablation (notebook section 3.2) loads its weights.

```bash
python train.py --train_for_scaling --save_dir results/models/ensemble_scaling
```

Key CLI arguments:

| Argument | Default | Description |
|---|---|---|
| `--num_models` | `5` | Number of ensemble members |
| `--epochs` | `5` | Training epochs per model |
| `--lr` | `2e-5` | Learning rate |
| `--model_name` | `xlm-roberta-base` | Base transformer model |
| `--batch_size` | `64` | Training batch size |
| `--beta` | `1.0` | Exponent for position score scaling |
| `--lora` | `False` | Use LoRA (flag, off by default) |
| `--save_dir` | `results/models/ensemble` | Output directory |
| `--train_for_scaling` | `False` | Train on entire dataset (no eval split) |

Expected runtime: **~2–4 hours** per run for 5 models × 5 epochs on an RTX 3090 / A100.

The three remaining model folders (`manifesto_ensemble_dl_10/`, `coalitionagree_ensemble_10/`, `tw_ensemble/`) are trained by specific cells **within the notebook** and do not require a separate `train.py` call.

---

### Step 4 — Wordfish Benchmarks (`04_r_prep.Rmd`)

> **Runs after Step 3a**, not before. Requires `data/py_outputs/manifesto.csv` (from §3.1 of `03a_main_analyses.ipynb`) and `data/py_outputs/tw_processed.csv` (from §3.4).

**What it does:**
1. **Wordfish — general** — country-by-country Wordfish; saves `data/r_outputs/wf_gen_all_countries.csv`.
2. **Wordfish — by topic** — per country × topic; saves `data/r_outputs/wf_topic_all_countries.csv`.
3. **Wordfish — environment protection** — CMP code 501; saves `data/r_outputs/wf_ep.csv`.
4. **Wordfish — welfare** — CMP codes 504–505; saves `data/r_outputs/wf_welfare.csv`.
5. **Wordfish — Twitter** — saves `data/r_outputs/wf_tw.csv`.

> **Note:** Wordfish models are run separately per country (and per topic) because Wordfish positions are not comparable across estimation contexts.

---
### Step 5 — Visualizations (`05_visualizations.Rmd`)

This R notebook produces all figures and tables in the paper. Run after completing all previous steps.

**Inputs loaded:**
- ContextScale scores: `results/datasets/ensemble_full_dataset.csv`, `data/py_outputs/cs_tw.csv`
- Wordfish scores: `data/r_outputs/wf_gen_all_countries.csv`, `data/r_outputs/wf_topic_all_countries.csv`
- Doc2Vec: `data/py_outputs/r&c_gen_party_election.csv`, `data/py_outputs/r&c_party_election_topic.csv`
- Classification metrics: all CSVs in `results/classification results/`

**Outputs:** figures and tables saved to `results/tabs and figs/`.

---

## Data Availability

All data and trained model weights needed to replicate the paper are archived on **[Harvard Dataverse (PSRM)](https://doi.org/10.7910/DVN/AFBN8X)**. The Dataverse deposit includes:

- Pre-generated manifesto corpus CSVs (`data/r_outputs/pulled_manifestoes.csv`, `pulled_manifestoes_test.csv`)
- All intermediate outputs (`data/r_outputs/`, `data/py_outputs/`)
- Trained ensemble model weights (`.safetensors`)
- Final scaled party position datasets (`results/datasets/`)

### Data sources

| Dataset | Source | Version | Notes |
|---------|--------|---------|-------|
| Manifesto corpus (text) | [manifesto-project.wzb.eu](https://manifesto-project.wzb.eu/) | MPDS2024a | Requires free API key; pre-generated CSVs on Dataverse |
| CMP metadata | [manifesto-project.wzb.eu](https://manifesto-project.wzb.eu/) | MPDS2024a | Included in `data/CMP/` |
| Coalition agreements | [COALITIONAGREE](https://doi.org/10.7910/DVN/XM5A08) | — | Included in `data/coalitionagree/` |
| Twitter ground-truth (MOTN) | [Bestvater & Monroe 2022](https://www.cambridge.org/core/journals/political-analysis/article/sentiment-is-not-stance-targetaware-opinion-classification-for-political-text-analysis/743A9DD62DF3F2F448E199BDD1C37C8D) | — | Included in `data/MOTN/` |

### Trained model weights

Model weights are available on PSRM Dataverse. Download and unzip into `results/models/`:


Weights can then be loaded directly:

```{python}
from safetensors.torch import load_file
state_dict = load_file("results/models/ensemble/model_ensemble_0.safetensors")
model.load_state_dict(state_dict)
```

See [REPLICATION_GUIDE.md](REPLICATION_GUIDE.md) for the full list of available model packages and detailed setup instructions.

---

## Citation

```bibtex
@article{nguyen2026contextscale,
  author  = {Nguyen, Hung H. V.},
  title   = {A Sentiment-Based Approach to Measuring Multidimensional Party Positions using Transformer},
  journal = {Political Science Research and Methods},
  year    = {forthcoming}
}
```
