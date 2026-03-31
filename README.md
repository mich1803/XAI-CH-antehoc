# XAI for Swiss Earthquakes (Diemtigen): EBM + 3-Segment Features

This repository contains full **pipeline-notebooks** for ante-hoc interpretable classification of **foreshocks vs aftershocks** in the Diemtigen sequence.

## What is kept (scope)

To keep the project focused and reproducible, the repo now uses:

- 1
  - **Model:** Explainable Boosting Machine (**EBM**)  
  - **Input representation:** **3-segment signal features** extracted from 3-component waveforms (E, N, Z)


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

- **Noise:** [0, 1) s
- **P window:** [1, 4) s
- **Coda:** [4, 15] s

For each segment and channel, the notebook extracts interpretable features and saves them as:

- `diemtigen_data/data_features.csv`
