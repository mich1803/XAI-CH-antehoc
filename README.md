# Interpretable Classification of Foreshocks vs Aftershocks -- Diemtigen Sequence

This repository explores **ante-hoc interpretable machine learning
models** for classifying seismic waveforms as **foreshocks or
aftershocks** using the **Diemtigen sequence dataset**.

The project focuses on designing **physically interpretable input
features** and combining them with **interpretable models** such as:

-   Logistic Regression (L1)
-   Generalized Additive Models (GAM)
-   Explainable Boosting Machines (EBM)

Unlike deep learning approaches (e.g., [CNN + SHAP](https://doi.org/10.1007/978-3-032-10185-3_25)), the goal is to
ensure that **interpretability is intrinsic to the model and input
representation**, not added afterwards.

The classification problem follows the definition of **ante-hoc
interpretability** described in:

-   Molnar et al., interpretable ML frameworks
-   https://www.sciencedirect.com/science/article/pii/S1389041724000378

------------------------------------------------------------------------

# Project Goal

Classify seismic events as:

-   **Foreshocks**
-   **Aftershocks**

using **3-component waveform recordings** while ensuring that:

1.  The **input features correspond to meaningful physical signal
    properties**
2.  The **model structure directly maps input features to predictions**
3.  The system avoids black-box feature extraction

This allows direct scientific interpretation such as:

> "High energy in the 10--20 Hz band during the early post-P window
> increases the probability of an aftershock."

------------------------------------------------------------------------

# Dataset Description

Waveforms have the following structure:

-   **3 components:** E, N, Z
-   **Sampling rate:** 100 Hz
-   **Duration:** 13 seconds
-   **Samples per channel:** 1300

The **P-wave arrival is fixed at 5 seconds**.

This allows the waveform to be divided into **physically meaningful
temporal segments**.

------------------------------------------------------------------------

# Waveform Segmentation Strategy

Using the fixed P arrival (5 seconds), the waveform is divided into **three interpretable temporal segments**:

| Segment | Time Range | Physical Meaning |
|---|---|---|
| Pre-P noise | [0,5) s | Background noise before the arrival |
| P + early signal | [5,8) s | Direct P arrival and early waveform |
| Coda | [8,13] s | Scattered energy and signal decay |

This segmentation allows the model to capture:

- signal-to-noise ratio between noise and arrival
- energy concentration during the P-wave window
- energy decay during the coda

These properties often differ between **foreshocks and aftershocks**.

------------------------------------------------------------------------

# Input Representations Tested

The project compares **three interpretable input strategies**.

## 1️⃣ Split Waveform Features

Features computed independently within each segment.

For each channel (E, N, Z) we compute:

### Time-Domain Features

-   RMS amplitude
-   Peak absolute amplitude
-   Signal energy
-   Standard deviation
-   Crest factor (peak / RMS)
-   Zero crossing rate

### Envelope Features

Envelope computed via Hilbert transform.

-   Maximum envelope
-   Envelope area
-   Envelope decay slope (coda windows)

### Frequency-Domain Features

Power spectral density (PSD) is computed and energy is integrated in
bands:

  Frequency Band
  ----------------
  1--5 Hz
  5--10 Hz
  10--20 Hz
  20--45 Hz

Additional features:

-   spectral centroid
-   spectral bandwidth

### Cross-Segment Ratios

Some particularly interpretable features include:

-   **Signal-to-noise proxy**

    RMS(5–8s) / RMS(0–5s)

-   **Coda/P energy ratio**

    Energy(8–13s) / Energy(5–8s)

-   **Vertical/Horizontal ratio**

    Energy(Z) / Energy(H)

These ratios describe how seismic energy evolves from the direct P-wave arrival into the coda phase.

------------------------------------------------------------------------

## 2️⃣ Low-Resolution Spectrogram Features

Instead of high-resolution spectrograms used by CNNs, we construct
**coarse interpretable time-frequency bins**.

Procedure:

-   Compute STFT
-   Use **1 second windows**
-   Use **10 Hz frequency bins**

Example bins:

  Frequency bins
  ----------------

0--10 Hz\
10--20 Hz\
20--30 Hz\
30--40 Hz\
40--50 Hz

For the **post-P window (5--13 s)**:

-   8 time bins
-   5 frequency bins

Total features per channel:

    8 × 5 = 40

Across three channels:

    120 spectrogram features

Each feature corresponds to:

> Energy in frequency band F during second T on channel C

This preserves interpretability.

------------------------------------------------------------------------

## 3️⃣ Hybrid Input (Split Features + Spectrogram)

This representation combines:

-   phase-based physical features
-   coarse time-frequency descriptors

The goal is to capture both:

-   physically interpretable signal properties
-   detailed temporal-frequency patterns

Expected feature count:

    ~150 features

------------------------------------------------------------------------

# Interpretable Models

## Logistic Regression (L1)

Model:

P(y=1\|x) = σ(β₀ + Σ βᵢxᵢ)

Properties:

-   linear additive model
-   coefficients directly represent feature influence

L1 regularization forces **sparse models**, selecting only the most
relevant features.

Interpretation example:

    β_energy_10_20Hz = +1.2

Meaning:

Increasing energy in the 10--20 Hz band increases the probability of an
aftershock.

------------------------------------------------------------------------

## Generalized Additive Models (GAM)

Model:

g(E\[y\]) = β₀ + Σ fᵢ(xᵢ)

Instead of linear coefficients, each feature has a **learned nonlinear
function**.

Advantages:

-   captures nonlinear thresholds
-   still interpretable feature-by-feature

Example interpretation:

    Aftershock probability increases rapidly when spectral centroid > 12 Hz

------------------------------------------------------------------------

## Explainable Boosting Machines (EBM)

EBM is an interpretable boosting method.

Model structure:

y = β₀ + Σ fᵢ(xᵢ) + Σ fᵢⱼ(xᵢ,xⱼ)

Properties:

-   additive feature effects
-   optional pairwise interactions
-   high accuracy while preserving interpretability

Visualization includes:

-   feature importance
-   shape functions
-   interaction heatmaps

------------------------------------------------------------------------

# Experimental Design

We evaluate the following combinations:

  Input Representation   Model
  ---------------------- ---------------------
  Split features         Logistic Regression
  Split features         GAM
  Split features         EBM
  Spectrogram            Logistic Regression
  Spectrogram            GAM
  Spectrogram            EBM
  Hybrid                 Logistic Regression
  Hybrid                 GAM
  Hybrid                 EBM

Metrics:

-   Accuracy
-   F1-score
-   ROC-AUC

Cross-validation should be used to avoid overfitting.

------------------------------------------------------------------------

# Expected Scientific Interpretation

The goal is not only classification accuracy but also **scientific
insights**, such as:

-   energy distribution differences between foreshocks and aftershocks
-   spectral shifts after the P arrival
-   coda decay differences
-   vertical vs horizontal energy ratios

These patterns can be directly derived from the learned feature effects.

------------------------------------------------------------------------

# Project Structure

Example repository structure:

    project/
    │
    ├── data/
    │   ├── raw_waveforms/
    │   └── processed/
    │
    ├── preprocessing/
    │   ├── segmentation.py
    │   ├── split_features.py
    │   └── spectrogram_features.py
    │
    ├── models/
    │   ├── logistic.py
    │   ├── gam.py
    │   └── ebm.py
    │
    ├── experiments/
    │   ├── run_split_features.py
    │   ├── run_spectrogram.py
    │   └── run_hybrid.py
    │
    └── README.md

------------------------------------------------------------------------

# Future Extensions

Possible improvements:

-   station-specific models
-   cross-station generalization experiments
-   inclusion of S-wave windows if picks are available
-   dimensionality reduction with interpretable methods (e.g., PCA
    loadings inspection)

------------------------------------------------------------------------

# References

Diemtigen sequence study:

https://doi.org/10.1029/2021GL093783

Interpretable ML concepts:

Molnar, *Interpretable Machine Learning*

Explainable Boosting Machine:

Microsoft InterpretML
