# XAI for Swiss Earthquakes (Diemtigen): EBM + 3-Segment Features

This repository contains a **single, coherent notebook workflow** for ante-hoc interpretable classification of **foreshocks vs aftershocks** in the Diemtigen sequence.

## What is kept (scope)

To keep the project focused and reproducible, the repo now uses only:

- **Model:** Explainable Boosting Machine (**EBM**)  
- **Input representation:** **3-segment signal features** extracted from 3-component waveforms (E, N, Z)

Removed from the workflow:

- Logistic Regression experiments
- Low-resolution spectrogram representation
- Hybrid (feature + spectrogram) representation

## Dataset assumptions

Input data is expected under `diemtigen_data/`:

- `events_mainshocks_foreshocks_aftershocks_15sec_23days.h5`
- `info_h5_events_mainshocks_foreshocks_aftershocks_15sec_23days.csv`

Waveform setup:

- 3 components: E, N, Z
- Sampling rate: ~120 Hz
- Duration: 15 s
- Fixed P arrival at 5 s

## 3-segment feature design

Each waveform is split into physically meaningful windows:

- **Noise:** [0, 5) s
- **P window:** [5, 8) s
- **Coda:** [8, 15] s

For each segment and channel, the notebook extracts interpretable features such as:

- RMS
- Energy
- Peak absolute amplitude
- Zero crossing rate

It also computes simple cross-segment ratios, e.g.:

- `p_rms / noise_rms` (signal-to-noise proxy)
- `coda_energy / p_energy`

The resulting table is saved as:

- `diemtigen_data/data_features.csv`

## Single notebook workflow

Run:

- `01_ebm_3segment_pipeline.ipynb`

This notebook performs end-to-end steps:

1. Load metadata and keep only foreshock/aftershock events
2. Build 3-segment feature table (`data_features.csv`)
3. Train/test split
4. Train additive EBM (`interactions=0`)
5. Evaluate with Accuracy, F1, ROC-AUC, confusion matrix
6. Interpret model via native EBM global importances and segment/channel aggregation

## Install

```bash
pip install -r requirements.txt
```

## Why this design

This repo targets **ante-hoc interpretability**: both features and model are inherently interpretable, so scientific interpretation can be made directly from learned effects rather than post-hoc explainers.
