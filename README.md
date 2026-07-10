# ROAMM

Machine learning code accompanying **ROAMM (Reading Observed At Mindless Moments)** — a large-scale, multimodal dataset of simultaneous EEG and eye-tracking recorded during naturalistic reading, with word-level mind-wandering (MW) annotations.

This repository contains the preprocessing, feature/batch generation, and modeling code used to build the benchmark tasks described in our ICML 2026 paper:

> **ROAMM: A Benchmark Dataset for Multimodal Human Attention Decoding and EEG-to-Text Modeling During Naturalistic Reading**
> Haorui Sun, Ardyn Olszko, Niharika Singh, David Jangraw
> *ICML 2026 — Poster, Hall A #414*
> [ICML poster page](https://icml.cc/virtual/2026/poster/60483) · [Dataset (OpenNeuro)](https://openneuro.org/datasets/ds007629)

## About ROAMM

ROAMM comprises roughly 50 hours of simultaneous 64-channel EEG and eye-tracking data collected from 44 participants during multi-page naturalistic reading. The dataset includes:

- Eye-tracking events (fixations, saccades, blinks) time-aligned to EEG
- Page-level comprehension scores
- Word-level mind-wandering (MW) labels, obtained via a retrospective self-report paradigm

Using this data, we define two benchmark tasks:

1. **Mind-wandering detection** — a standardized leave-one-subject-out (LOSO) evaluation protocol for classifying MW from EEG (and eye-tracking) signals, achieving up to 0.609 AUROC with supervised models.
2. **EEG-to-text decoding** — decoding read text from EEG, trained on non-MW segments, showing that decoding performance degrades when MW-labeled segments are included — demonstrating attention-related degradation in brain-to-language decoding during naturalistic reading.

Raw EEG/eye-tracking data are available upon request; the full processed dataset is hosted on OpenNeuro (see link above).

## Repository structure

| Path | Description |
|---|---|
| `batches/` | Scripts/configs for generating windowed, subject/run-level batches of aligned EEG + eye-tracking data used for model training and evaluation |
| `eeg2text/` | Code for the EEG-to-text decoding task, including training on non-MW segments and evaluating the effect of MW-labeled segments on decoding performance |
| `eegfm/` | Code related to EEG foundation-model features/embeddings used as inputs to downstream MW detection and decoding models |
| `notebooks/` | Jupyter notebooks for exploratory data analysis, data validation, and result visualization |
| `res/` | Supporting resources (e.g., reference/result files used by the analysis and modeling scripts) |
| `scripts/` | Standalone preprocessing and pipeline scripts (e.g., dataset preparation, alignment, LOSO evaluation) |
| `roamm_environment.yml` | Conda environment specification |
| `requirements.txt` | Pip dependency list |
| `loso_subject_metric_histograms_512win.pdf` | Per-subject LOSO evaluation metric distributions (512-sample window setting) referenced in the paper |
| `LICENSE` | MIT License |

> Note: this table reflects the top-level layout of the repo at the time of writing. See each subfolder for further details/READMEs where available.

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/GlassBrainLab/roamm_ml.git
cd roamm_ml
```

### 2. Set up the environment

Using conda:

```bash
conda env create -f roamm_environment.yml
conda activate roamm  # adjust to the environment name defined in the yml file
```

Or using pip:

```bash
pip install -r requirements.txt
```

### 3. Get the data

The processed ROAMM dataset is hosted on OpenNeuro:

- **Dataset:** https://openneuro.org/datasets/ds007629

Raw (unprocessed) EEG and eye-tracking recordings are not hosted online due to size, but are available upon request — see the dataset page for contact details.

### 4. Reproduce the benchmarks

- Use the scripts in `batches/` and `scripts/` to build subject/run-level, EEG–eye-tracking-aligned batches from the downloaded dataset.
- Use `eegfm/` for foundation-model-derived features/embeddings, and `eeg2text/` for the EEG-to-text decoding pipeline.
- The `notebooks/` folder contains examples for data loading, validation, and inspecting results (e.g., the LOSO MW-detection metrics summarized in `loso_subject_metric_histograms_512win.pdf`).

## Citation

If you use this code or the ROAMM dataset, please cite:

```bibtex
@inproceedings{sun2026roamm,
  title     = {ROAMM: A Benchmark Dataset for Multimodal Human Attention Decoding and EEG-to-Text Modeling During Naturalistic Reading},
  author    = {Sun, Haorui and Olszko, Ardyn and Singh, Niharika and Jangraw, David},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

(Please update this entry with the final proceedings citation once available on PMLR.)

## License

This project is released under the [MIT License](LICENSE).

## Contact

For questions about the dataset, benchmark tasks, or this codebase, please open an issue in this repository or refer to the contact information on the [OpenNeuro dataset page](https://openneuro.org/datasets/ds007629).
