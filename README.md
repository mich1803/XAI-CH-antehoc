# XAI for Swiss Earthquakes (Diemtigen)

This repository contains reproducible notebook pipelines for **foreshock vs aftershock** classification on the Diemtigen sequence, with a focus on **ante-hoc interpretability**.

## Available notebooks

### 1) `ebm_on_signal_features.ipynb`
Ante-hoc tabular baseline using:
- **Model:** Explainable Boosting Machine (EBM)
- **Input representation:** handcrafted **3-segment signal features** from 3-component waveforms (E, N, Z)

### 2) `attention_transformer_antehoc.ipynb`
Ante-hoc deep-learning baseline using:
- **Model:** lightweight token-based Transformer
- **Input representation:** overlapping waveform tokens (windowed raw signal)
- **Interpretability outputs:**
  - token-level CLS attention inspection
  - **attention faithfulness** diagnostics via removal/insertion curves
  - optional attention entropy regularization (sparsity pressure)

> Important: attention is treated as a *candidate explanation signal* and always paired with faithfulness checks.

## Dataset assumptions

Input data is expected under `diemtigen_data/`:
- `events_mainshocks_foreshocks_aftershocks_15sec_23days.h5`
- `info_h5_events_mainshocks_foreshocks_aftershocks_15sec_23days.csv`

Waveform setup (expected by both notebooks):
- 3 components: E, N, Z
- Sampling rate: ~120 Hz
- Duration: 15 s
- Fixed P arrival at 5 s

## EBM feature segmentation (notebook 1)

For the EBM pipeline, each waveform is split into physically meaningful windows:
- **Noise:** [0, 1) s
- **P window:** [1, 4) s
- **Coda:** [4, 15] s

Per-segment/per-channel features are exported to:
- `diemtigen_data/data_features.csv`

## Environment setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies include:
- data/science stack: `numpy`, `pandas`, `scipy`, `scikit-learn`, `h5py`
- interpretability: `interpret`, `pygam`
- deep learning + plotting: `torch`, `tqdm`, `matplotlib`, `seaborn`

## Running the notebooks

1. Place dataset files under `diemtigen_data/`.
2. Open Jupyter Lab/Notebook in this repo.
3. Run either notebook end-to-end:
   - `ebm_on_signal_features.ipynb`
   - `attention_transformer_antehoc.ipynb`

For desktop usage, the transformer notebook automatically uses CUDA when available and falls back to CPU.
