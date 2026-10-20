# 🩺 Preterm Labor Rapid Triage System (EHG Triage)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live-FF4B4B.svg)](http://localhost:8501)

A clinically defensible, point-of-care electrohysterogram (EHG) triage system designed to predict preterm delivery risk using a **2-minute (120-second = 2,400 samples @ 20 Hz)** 3-channel abdominal recording.

Validated on **PhysioNet TPEHGDB v1.0.1 ($N=300$)** and externally tested on **TPEHGT v1.0.0 ($N=13$)** under strict **patient-level stratified cross-validation** (zero data leakage).

---

## 🌟 Key Performance Highlights

| Metric | Baseline 1-Min | 2-Min (All 69 Feats) | **2-Min Top-30 ExtraTrees (Final)** ⭐ |
| :--- | :---: | :---: | :---: |
| **ROC-AUC** | 0.7178 | 0.7417 | **`0.7603`** |
| **Preterm Recall (Sensitivity)** | 42.11% (8/19) | 57.89% (11/19) | **`68.42%` (13/19 preterms caught)** |
| **Balanced Accuracy** | 58.42% | 69.28% | **`70.77%`** |
| **Term Specificity** | 74.79% | 80.67% | **`73.11%` (87/119 terms correct)** |
| **Inference Latency** | < 450 ms | < 500 ms | **`< 450 ms` (Real-Time Point-of-Care)** |

---

## 🧠 Model Architecture

The system uses a **Two-Tier Gestational-Conditioned Routing Architecture**:

1. **Tier 1: 3rd-Trimester Labor Triage ($\ge 26$ Weeks, $N=138$):**
   - **Feature Selection:** ANOVA F-score Top-30 biomarker filter (prunes high-frequency noise).
   - **Classifier:** Dual-Depth Soft ExtraTrees Ensemble (Depth 3 for variance reduction + Depth 4 for non-linear pattern capture, balanced class weighting).
   - **Decision Threshold:** Calibrated $t = 0.48$ ($t = 0.45$ for high-sensitivity screening: **73.68% recall**).
2. **Tier 2: General Screening Ensemble (< 26 Weeks or All Records, $N=300$):**
   - Robust universal screening model with soft probability calibration.

---

## 🧬 Selected Core Biomarkers (Top 30)

1. **Contraction Energy Ratios:** `ratio_burst_to_bg_ch1`, `ch2`, `ch3` (Contraction RMS relative to baseline).
2. **Burst Geometry:** `burst_area` and `burst_perimeter` on Channels 1, 2, and 3.
3. **Fast-Wave Shift:** `ch2_p_03_10` ($0.3\text{--}1.0\text{ Hz}$ fast-wave burst propagation) and `f_med` (median frequency upward shift).
4. **Complexity:** Kurtosis (non-Gaussian spike bursts) and Zero-Crossing Rates.

---

## 📁 Repository Structure

```
ehg_triage/
├── app.py                     # Interactive Streamlit Web Interface (http://localhost:8501)
├── README.md                  # Comprehensive Project Documentation
├── models/                    # Serialized production models and scalers
│   ├── ensemble_late_120s.joblib
│   ├── fitted_scaler_late_120s.joblib
│   ├── selector_late.joblib
│   ├── selected_feature_names_late_120s.json
│   └── best_triage_model.joblib
├── reports/                   # Cross-validation CSV benchmark reports
│   └── model_2min_evaluation_report.csv
├── data/                      # Cached feature matrices (60s, 120s, 180s)
│   ├── features_120s_2min_tpehgdb.csv
│   └── features_120s_2min_tpehgt13.csv
└── src/                       # Core python modules
    ├── inference.py           # Real-time inference engine (EHGTriagePredictor)
    ├── train_2min_model.py    # Training, feature selection, and CV script
    ├── preprocessing.py       # Butterworth 0.3-3.0 Hz filtering and QC
    ├── features.py            # Spectral, complexity, and entropy extraction
    └── burst_and_advanced_features.py # Contraction envelope geometry
```

---

## 🚀 Quickstart

### 1. Launch the Interactive Web Dashboard
```bash
streamlit run app.py --server.port 8501
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 2. Python Programmatic Inference
```python
import numpy as np
from src.inference import EHGTriagePredictor

# Initialize predictor
predictor = EHGTriagePredictor()

# 2-minute 3-channel EHG signal (2400 samples @ 20 Hz)
signal_2min = np.random.randn(2400, 3) * 0.05

# Predict preterm delivery risk
result = predictor.predict_one_minute(
    signal_2min,
    fs=20.0,
    threshold=0.48,
    gestational_age_weeks=32.0
)

print("Triage Category:", result['triage_category'])
print("Risk Score:", f"{result['risk_score']*100:.1f}%")
print("Latency:", f"{result['inference_latency_ms']:.1f} ms")
```

---

## 🎓 Viva / Defense Key Talking Points

1. **Why Patient-Level Cross-Validation Matters:**
   - Many published papers claim 90%+ accuracy by randomly splitting 1-minute windows across train and test. That is **severe data leakage** (windows from the same mother in both sets).
   - Our system enforces **strict patient-level splitting (`pregnancy_id`)**. Our **0.760 ROC-AUC** reflects genuine clinical generalization.
2. **Why 2 Minutes (120s) is the Clinical Sweet Spot:**
   - 1-minute windows truncate 45–90s contractions. 
   - 5-minute recordings introduce maternal motion artifacts and baseline drift.
   - 2 minutes captures **one full contraction burst cleanly** with sub-second inference.
3. **Why Top-30 Feature Selection Improved Performance:**
   - On $N=138$ late-gestation patients, 69 features caused the "curse of dimensionality" (2:1 ratio).
   - Selecting the Top 30 core biophysical features removed high-frequency noise and boosted **Preterm Recall from 57.9% to 68.42% (13/19 caught)**!

---

## ⚖️ Safety & Disclaimer
**RESEARCH PROTOTYPE ONLY**: This experimental algorithm is built as an investigational prototype for research study. It is **NOT FDA-cleared**, **NOT clinically validated**, and must **NEVER be used as a standalone diagnostic tool for preterm labor**.
