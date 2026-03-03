# ContextScale

Replication repository for:

> Nguyen, Hung H. V. "A Sentiment-Based Approach to Measuring Multidimensional Party Positions using Transformer ." *Political Science Research and Methods* (2026). 

**ContextScale** is a transformer-based framework for scaling party positions from political texts. It uses a deep ensemble of XLM-RoBERTa models with a shared attention architecture (`ContextScalePrediction`) to simultaneously predict topic categories and sentiment/stance from quasi-sentence level manifesto data, producing uncertainty-quantified party position scores. The approach is validated against Wordfish, CHES expert survey data, and Doc2Vec (Rheault & Cochrane 2020), and is extended to coalition agreements and Twitter data.

Final datasets and trained model weights are shared via [PSRM Dataverse](https://dataverse.harvard.edu/dataverse/PSRM).

---

## Repository Structure

```
ContextScale/
├── 01_manifesto_pull.Rmd      # Step 1 (R): Pull manifesto data from CMP API
├── 02_r_prep.Rmd              # Step 2 (R): Wordfish models & data preparation
├── 03_main_analyses.ipynb     # Step 3 (Python): Model training, ablation, scaling
├── 04_visualizations.Rmd      # Step 4 (R): All figures and tables in the paper
├── train.py                   # Standalone training script (CLI interface)
├── utils/
│   ├── functions.py           # Data preprocessing, training/eval loops, scaling
│   ├── models.py              # ContextScalePrediction model definition
│   ├── uncertainty.py         # Deep ensemble training and inference
│   └── legacy.py              # Archive of earlier code
├── data/
│   ├── CMP/                   # Comparative Manifesto Project metadata
│   ├── ches/                  # CHES expert survey data
│   ├── MOTN/                  # Twitter ground-truth datasets
│   ├── r_outputs/             # Outputs from R scripts (manifesto CSVs, Wordfish)
│   ├── py_outputs/            # Outputs from Python scripts (Doc2Vec, CS scores)
│   └── temps/                 # Cached preprocessed data (auto-generated)
└── results/
    ├── classification results/ # F1/precision/recall tables per model variant
    ├── datasets/              # Final scaled datasets
    ├── models/                # Saved model weights (.safetensors)
    └── tabs and figs/         # Figures and tables for the paper
```

---

## Replication Instructions

The analysis pipeline consists of four sequential steps. Steps 1–2 and Step 4 run in **R**; Step 3 runs in **Python**.

### Prerequisites

**R packages** (install via `pacman::p_load` or `install.packages`):
```r
pacman::p_load(tidyverse, manifestoR, quanteda, quanteda.textmodels,
               ggplot2, viridis, ggsci, ggpubr, ragg, scales, gt,
               rstatix, gridExtra, openxlsx, countrycode, scico, caret)
```

**Python packages:**
```bash
pip install torch transformers datasets scikit-learn pandas numpy \
            gensim spacy nltk safetensors
```

A CUDA-capable GPU (≥8 GB VRAM) is strongly recommended for Step 3.

**Manifesto API key:** Register at [manifesto-project.wzb.eu](https://manifesto-project.wzb.eu/) and place your API key in `manifesto_apikey.txt` in the root folder.

---

### Step 1 — Pull Manifesto Data (`01_manifesto_pull.Rmd`)

This R script uses the `manifestoR` package to download annotated manifesto quasi-sentences from the Comparative Manifesto Project (CMP) API.

- **Training countries** (17 Western European democracies): Austria, Belgium, Denmark, Finland, France, Germany, Greece, Iceland, Ireland, Italy, Netherlands, Norway, Portugal, Spain, Sweden, Switzerland, United Kingdom
- **Test countries** (out-of-sample, unseen languages): Czech Republic, Japan, Turkey, Russia, South Korea

**Outputs:**
- `data/r_outputs/pulled_manifestoes.csv` — training corpus
- `data/r_outputs/pulled_manifestoes_test.csv` — out-of-sample test corpus

---

### Step 2 — R Data Preparation (`02_r_prep.Rmd`)

This R script prepares the Wordfish benchmarks and supplementary datasets used for validation.

**What it does:**
1. **CHES data cleaning** — processes the Chapel Hill Expert Survey (1999–2019) and saves `data/r_outputs/ches_cleaned.csv`.
2. **Wordfish — general** — runs country-by-country Wordfish models on the full manifesto corpus; saves `data/r_outputs/wf_gen_all_countries.csv`.
3. **Wordfish — by topic** — runs Wordfish separately for each country × topic combination (Economics, Labour & Social Welfare, Immigration, European Integration, Environment–Growth); saves `data/r_outputs/wf_topic_all_countries.csv`.
4. **Wordfish — environment protection** — runs Wordfish on CMP code 501 only; saves `data/r_outputs/wf_ep.csv`.
5. **Wordfish — welfare** — runs Wordfish on CMP codes 504–505; saves `data/r_outputs/wf_welfare.csv`.
6. **Wordfish — Twitter** — runs Wordfish on the processed Twitter dataset; saves `data/r_outputs/wf_tw.csv`.
7. **Coalition agreements** — prepares coalition contract texts for subsequent scaling.

> **Note:** Wordfish models are run separately per country (and per topic) because Wordfish positions are not comparable across estimation contexts.

---

### Step 3 — Main Analyses (`03_main_analyses.ipynb`)

This is the core Python notebook. Run cells sequentially. A GPU is required for training; all other cells (loading, inference, scaling) can run on CPU but will be slower.

The notebook is divided into the following sections:

#### 3.1 Data Preparation
- Loads `data/r_outputs/pulled_manifestoes.csv`
- Applies `sentiment_code` and `topic_code` to assign labels from `cmp_code`
- Groups short quasi-sentences into longer sequences using `group_texts` (max 5x grouping factor, to fit within the 512-token limit)
- Saves preprocessed data to `data/temps/manifesto.csv` and `data/temps/manifesto_regrouped.csv`
- Tokenizes data and builds stratified train/eval/test splits

#### 3.2 Deep Ensemble Training & Inference
Training is delegated to `train.py` for long-running sessions (see below). The notebook handles:
- Loading trained ensemble weights from `results/models/ensemble/`
- Running ensemble inference on the test set (mean position scores + uncertainty estimates)
- Saving results to `results/datasets/ensemble_results_test.pkl` and `results/datasets/ensemble_full_dataset.csv`

#### 3.3 Ablation — Architecture Comparison
Trains and evaluates four model variants on the same dataset:

| Variant | Description | Save path |
|---|---|---|
| `base` | Standard XLM-RoBERTa + two independent heads | `results/models/manifesto_ContextScalePrediction_base/` |
| `sf` | Simple (unidirectional) information flow between heads | `results/models/manifesto_ContextScalePrediction_sf/` |
| `sa` | **Shared attention** (proposed architecture) | `results/models/manifesto_ContextScalePrediction_sa/` |
| `dg` | Dynamic gating between heads | `results/models/manifesto_ContextScalePrediction_dg/` |

Classification metrics (F1, precision, recall, accuracy per topic/sentiment class) are saved to `results/classification results/`.

#### 3.4 Validity Checks
- Tests the trained model on the out-of-sample countries (Czech Republic, Japan, Turkey, Russia, South Korea) to assess cross-lingual transfer.

#### 3.5 Transfer to Twitter Data
- Loads three ground-truth Twitter datasets from `data/MOTN/` (Trump stance, Kavanaugh, Women's March)
- Recodes stance labels to left/right using `recode_tw`
- Transfers weights from the manifesto-trained model and fine-tunes on 20% of Twitter data
- Saves adapted ensemble to `results/models/tw_ensemble/` and results to `data/py_outputs/cs_tw.csv`

#### 3.6 Doc2Vec Scaling (Rheault & Cochrane 2020 replication)
- Trains country-level Doc2Vec models (vector size 500, PCA to 2D) as a baseline comparison
- General, topic-specific, environment-protection, welfare, and Twitter variants
- Outputs saved to `data/py_outputs/r&c_*.csv`

#### 3.7 Full Dataset Scaling (Released Model)
- Combines training + test country manifestos and coalition agreements
- Retrains the final model and saves weights to `results/models/contextscale_full_released/`
- Computes position scores for all manifesto quasi-sentences and aggregates to country–party–election–topic level
- **Released datasets:** `results/datasets/contextscale_manifesto_dataset.csv` and `results/datasets/contextscale_coalition_dataset.csv`

---

### Standalone Training (`train.py`)

For long training runs, use `train.py` instead of the notebook training cells. It replicates the notebook's data pipeline exactly and supports persistent `tmux`/`screen` sessions.

```bash
# Default (5 models, 5 epochs, xlm-roberta-large)
python train.py

# Custom configuration
python train.py --num_models 5 --epochs 5 --lr 2e-5 --model_name xlm-roberta-base \
                --batch_size 16 --beta 1.0 --lora True \
                --save_dir results/models/ensemble_splits
```

Key CLI arguments:

| Argument | Default | Description |
|---|---|---|
| `--num_models` | 5 | Number of ensemble members |
| `--epochs` | 5 | Training epochs per model |
| `--lr` | 2e-5 | Learning rate |
| `--model_name` | `xlm-roberta-large` | Base transformer model |
| `--batch_size` | 16 | Training batch size |
| `--beta` | 1.0 | Exponent for position score scaling |
| `--lora` | True | Use LoRA for parameter-efficient fine-tuning |
| `--save_dir` | `results/models/ensemble_splits` | Output directory |

Expected runtime: ~2–4 hours for 5 models × 5 epochs on a modern GPU.

For persistent training sessions:
```bash
tmux new-session -d -s training
tmux attach-session -t training
python train.py --num_models 5 --epochs 5
# Detach with Ctrl+B, D
```

---

### Step 4 — Visualizations (`04_visualizations.Rmd`)

This R notebook produces all figures and tables in the paper. Run after completing Steps 1–3.

**Inputs loaded:**
- ContextScale scores: `data/py_outputs/ensemble_full_dataset.csv`, `data/py_outputs/cs_tw.csv`
- Wordfish scores: `data/r_outputs/wf_gen_all_countries.csv`, `data/r_outputs/wf_topic_all_countries.csv`
- CHES: `data/r_outputs/ches_cleaned.csv`
- Doc2Vec: `data/py_outputs/r&c_gen_party_election.csv`, `data/py_outputs/r&c_party_election_topic.csv`
- Classification metrics: all CSVs in `results/classification results/`

**Outputs:** figures and tables saved to `results/tabs and figs/`.

---

## Data Availability

The Manifesto corpus requires a free API key from [manifesto-project.wzb.eu](https://manifesto-project.wzb.eu/). The CHES dataset is available at [chesdata.eu](https://www.chesdata.eu/). The MOTN Twitter datasets are from the paper **Sentiment is Not Stance: Target-Aware Opinion Classification for Political Text Analysis** (https://www.cambridge.org/core/journals/political-analysis/article/sentiment-is-not-stance-targetaware-opinion-classification-for-political-text-analysis/743A9DD62DF3F2F448E199BDD1C37C8D).

Final scaled datasets (party position scores) are available on [PSRM Dataverse](https://dataverse.harvard.edu/).

---

## Citation

```bibtex
@article{nguyen2023contextscale,
  author  = {Nguyen, Hung H. V.},
  title   = {An Improved Framework for Scaling Party Positions from Texts with Transformer},
  journal = {Political Science Research and Methods},
  year    = {forthcoming},
  note    = {Preprint: SocArXiv. \url{https://doi.org/10.31235/osf.io/8sha3}}
}
```
