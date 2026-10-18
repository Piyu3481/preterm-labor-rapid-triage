# One-Minute EHG Preterm-Delivery Risk Triage Research Prototype

## 1. Project Overview & Clinical Objective
This research prototype investigates whether a **60-second, three-channel electrohysterogram (EHG / uterine EMG)** segment can classify pregnancies according to recorded preterm (delivery at $\le 37$ weeks) versus term outcome.

> [!WARNING]
> **RESEARCH PROTOTYPE DISCLAIMER:**
> This algorithm is an investigational risk-stratification proxy, **NOT** a clinically validated diagnostic tool. It does **NOT** diagnose active preterm labor and has not been cleared by regulatory authorities.

---

## 2. Phase 1 Data Audit & Critical Findings

We audited two open PhysioNet databases collected at the University Medical Centre Ljubljana:
1. **TPEHGDB v1.0.1** (*Term-Preterm EHG Database*): 300 records from 300 pregnant women.
   - **Term ($>37$ wks):** 262 records (87.3%)
   - **Preterm ($\le 37$ wks):** 38 records (12.7%)
   - **Sampling Rate:** 20.0 Hz (16-bit ADC, $0.0-5.0\text{ Hz}$ analog hardware filter).
2. **TPEHGT v1.0.0** (*Term-Preterm EHG Dataset with Tocogram*): 31 records.
   - **Non-pregnant controls:** 5 records (`tpehgt_n001` to `n005`) $\rightarrow$ **strictly excluded**.
   - **Preterm:** 13 records from 8 pregnancies.
   - **Term:** 13 records from 10 pregnancies.

### Cross-Dataset Overlap Discovery
Cross-dataset signal correlation analysis revealed that **TPEHGT is not an independent dataset**:
- **8 records in TPEHGT are identical recordings to TPEHGDB records** ($\text{correlation} \ge 0.9994\text{ to }1.0000$):
  - `tpehgt_p003` $\leftrightarrow$ `tpehg1734`
  - `tpehgt_p006` $\leftrightarrow$ `tpehg1741`
  - `tpehgt_p008` $\leftrightarrow$ `tpehg1751`
  - `tpehgt_p010` $\leftrightarrow$ `tpehg1756`
  - `tpehgt_t008` $\leftrightarrow$ `tpehg1732`
  - `tpehgt_t009` $\leftrightarrow$ `tpehg1735`
  - `tpehgt_t011` $\leftrightarrow$ `tpehg1746`
  - `tpehgt_t012` $\leftrightarrow$ `tpehg1747`
- **5 additional TPEHGT records belong to the same pregnancies** (`p002`, `p007`, `p009`, `p011`, `t007`).
- **Label Conflict on Patient `tpehgt_p006` / `tpehg1741` Resolution:** Delivery occurred between 36 and 37 weeks; rounded up to 37 (Term) in TPEHGDB vs recorded as 36 (Preterm) in TPEHGT. **Authoritative Resolution:** We designate TPEHGDB as authoritative since it is the primary benchmark reference dataset, standardizing `tpehgt_p006` / `tpehg1741` as **Term (0)**.

The complete manifest is saved to [`data/record_manifest.csv`](data/record_manifest.csv).

---

## 3. Signal Pipeline & 60-Second Segment Preprocessing

### Channel Standardization
All models operate on the standardized 3 bipolar abdominal EHG channels:
- **$S_1$ (Upper horizontal):** $E_2 - E_1$ (TPEHGDB ch 0, TPEHGT ch 0)
- **$S_2$ (Right vertical):** $E_2 - E_3$ (TPEHGDB ch 4, TPEHGT ch 2)
- **$S_3$ (Lower horizontal):** $E_4 - E_3$ (TPEHGDB ch 8, TPEHGT ch 4)

### Bandpass Filtering & Transient Protection
- Standard 4-pole Butterworth digital bandpass filter ($0.3 - 3.0\text{ Hz}$), applied bi-directionally with zero-phase distortion (`scipy.signal.filtfilt`).
- The documented **180-second filter transients** at the beginning and end of recordings are excluded.
- Segment extraction policy (`center`): Extracts the central 60-second window ($1200\text{ samples}$ at $20\text{ Hz}$), well away from boundary transients.

---

## 4. Model Benchmarks: Before vs. After Sensitivity Collapse Fix

### The Sensitivity Collapse Issue (Original Phase 4)
In the initial uncalibrated evaluation using default decision threshold 0.50, non-linear models (SVM, Random Forest, XGBoost) completely collapsed into predicting the majority class ("Term"), reporting **0.000 Sensitivity**.

### The Fix Implemented
1. **Inside-Fold SMOTE:** Applied SMOTE oversampling ($k=3$) strictly inside the training fold (never on validation or test data).
2. **Leak-Free Youden's J Threshold Tuning:** Tuned decision thresholds per model inside each training fold using an inner 3-fold cross-validation maximizing $J = \text{Sensitivity} + \text{Specificity} - 1$.
3. **Confirmed Fold-Level Class Weighting:** Enforced `class_weight='balanced'` and `scale_pos_weight` inside each cross-validation fold.

### Side-by-Side Before vs. After Comparison (5-Fold Stratified Patient CV, TPEHGDB Cohort)

| Model | Setup | Decision Threshold | Accuracy | Balanced Accuracy (95% CI) | Sensitivity / Recall (95% CI) | Specificity (95% CI) | F1-Score (95% CI) | ROC-AUC (95% CI) | PR-AUC (95% CI) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline** | Default | 0.50 | 0.873 | 0.500 [0.500–0.500] | 0.000 [0.000–0.000] | 1.000 [1.000–1.000] | 0.000 [0.000–0.000] | 0.500 [0.500–0.500] | 0.127 [0.090–0.167] |
| **SVM (RBF Kernel)** | **Original** | 0.50 | 0.873 | 0.500 [0.500–0.500] | **0.000 [0.000–0.000]** | 1.000 [1.000–1.000] | 0.000 [0.000–0.000] | 0.443 [0.351–0.543] | 0.114 [0.082–0.171] |
| **SVM (RBF Kernel)** | **Enhanced** | **0.49** | **0.763** | **0.471 [0.442–0.548]** | **0.079 [0.024–0.219]** | **0.863 [0.834–0.911]** | **0.078 [0.026–0.209]** | **0.476 [0.387–0.575]** | **0.121 [0.082–0.186]** |
| **Random Forest** | **Original** | 0.50 | 0.870 | 0.498 [0.494–0.500] | **0.000 [0.000–0.000]** | 0.996 [0.988–1.000] | 0.000 [0.000–0.000] | 0.413 [0.323–0.511] | 0.109 [0.078–0.170] |
| **Random Forest** | **Enhanced** | **0.42** | **0.507** | **0.459 [0.414–0.527]** | **0.395 [0.023–0.219]** | **0.523 [0.780–0.868]** | **0.169 [0.022–0.182]** | **0.430 [0.336–0.529]** | **0.108 [0.075–0.159]** |
| **XGBoost** | **Original** | 0.50 | 0.740 | 0.424 [0.401–0.445] | **0.000 [0.000–0.000]** | 0.847 [0.802–0.891] | 0.000 [0.000–0.000] | 0.436 [0.354–0.515] | 0.106 [0.076–0.145] |
| **XGBoost** | **Enhanced** | **0.41** | **0.457** | **0.464 [0.367–0.530]** | **0.474 [0.189–0.500]** | **0.454 [0.496–0.618]** | **0.181 [0.079–0.231]** | **0.428 [0.331–0.525]** | **0.108 [0.075–0.155]** |
| **Logistic Regression** | **Original** | 0.50 | 0.667 | 0.539 [0.454–0.624] | 0.368 [0.210–0.533] | 0.710 [0.653–0.764] | 0.219 [0.126–0.323] | 0.582 [0.489–0.671] | 0.194 [0.118–0.306] |
| **Logistic Regression** | **Enhanced** | **0.44** | **0.607** | **0.516 [0.369–0.517]** | **0.395 [0.077–0.351]** | **0.637 [0.618–0.727]** | **0.203 [0.045–0.206]** | **0.494 [0.404–0.585]** | **0.162 [0.101–0.277]** |

---

## 5. Separate Evaluation Regimens

### Table 1: TPEHGT-Only Standalone Cross-Validation (26 Pregnant Records)
| Model | Accuracy | Balanced Accuracy (95% CI) | Sensitivity / Recall (95% CI) | Specificity (95% CI) | F1-Score (95% CI) | ROC-AUC (95% CI) | PR-AUC (95% CI) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline** | 0.538 | 0.500 [0.500–0.500] | 0.000 [0.000–0.000] | 1.000 [1.000–1.000] | 0.000 [0.000–0.000] | 0.500 [0.500–0.500] | 0.462 [0.269–0.654] |
| **Logistic Regression** | 0.269 | 0.268 [0.077–0.390] | 0.250 [0.000–0.400] | 0.286 [0.069–0.559] | 0.240 [0.000–0.364] | 0.268 [0.057–0.479] | 0.417 [0.229–0.649] |
| **SVM (RBF Kernel)** | 0.615 | 0.619 [0.500–0.714] | 0.667 [1.000–1.000] | 0.571 [0.000–0.429] | 0.615 [0.483–0.842] | 0.699 [0.492–0.877] | 0.720 [0.464–0.905] |
| **Random Forest** | 0.423 | 0.411 [0.164–0.506] | 0.250 [0.000–0.423] | 0.571 [0.200–0.742] | 0.286 [0.000–0.400] | 0.295 [0.105–0.494] | 0.372 [0.219–0.615] |

> [!WARNING]
> **Sample Size & Statistical Power Limitation:**
> With only 26 pregnant records from 18 distinct pregnancies, each cross-validation fold contains merely 3 to 5 records. A single misclassified patient shifts fold metrics by 20 percentage points. This causes extremely wide bootstrap confidence intervals (e.g. ROC-AUC spanning 0.49 to 0.88 for SVM). Standalone TPEHGT CV is statistically underpowered and should be treated purely as an exploratory feasibility check, not a validated benchmark.

### Table 2: Combined Clean Cohort (313 Records: 300 DB + 13 Independent TPEHGT)
All 13 overlapping TPEHGT records were strictly excluded from this cohort to eliminate duplicate signal leakage:
| Model | Accuracy | Balanced Accuracy (95% CI) | Sensitivity / Recall (95% CI) | Specificity (95% CI) | F1-Score (95% CI) | ROC-AUC (95% CI) | PR-AUC (95% CI) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline** | 0.863 | 0.500 [0.500–0.500] | 0.000 [0.000–0.000] | 1.000 [1.000–1.000] | 0.000 [0.000–0.000] | 0.500 [0.500–0.500] | 0.137 [0.099–0.176] |
| **Logistic Regression** | 0.665 | 0.473 [0.365–0.504] | 0.209 [0.091–0.341] | 0.737 [0.595–0.714] | 0.146 [0.054–0.200] | 0.435 [0.350–0.526] | 0.131 [0.088–0.208] |
| **SVM (RBF Kernel)** | 0.757 | 0.468 [0.433–0.550] | 0.070 [0.045–0.262] | 0.867 [0.786–0.878] | 0.073 [0.043–0.229] | 0.471 [0.385–0.570] | 0.132 [0.094–0.194] |
| **Random Forest** | 0.610 | 0.442 [0.419–0.513] | 0.209 [0.022–0.191] | 0.674 [0.787–0.874] | 0.129 [0.021–0.171] | 0.402 [0.327–0.491] | 0.114 [0.082–0.177] |
| **XGBoost** | 0.383 | 0.398 [0.352–0.498] | 0.419 [0.154–0.413] | 0.378 [0.504–0.627] | 0.157 [0.073–0.208] | 0.390 [0.301–0.477] | 0.120 [0.083–0.190] |

### Table 3: TPEHGDB-to-TPEHGT External Transfer Evaluation (Train 300 DB $\rightarrow$ Test 13 Indep TPEHGT)
Trained on all 300 TPEHGDB records with inside-fold SMOTE and threshold calibration, then evaluated on the 13 non-overlapping independent pregnant records in TPEHGT:
| Model | Tuned Threshold | Accuracy | Balanced Accuracy (95% CI) | Sensitivity / Recall (95% CI) | Specificity (95% CI) | F1-Score (95% CI) | ROC-AUC (95% CI) | PR-AUC (95% CI) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.15 | 0.385 | 0.312 [0.500–0.500] | 0.000 [0.000–0.000] | 0.625 [1.000–1.000] | 0.000 [0.000–0.000] | 0.400 [0.097–0.720] | 0.363 [0.169–0.706] |
| **SVM (RBF Kernel)** | 0.85 | 0.615 | 0.500 [0.306–0.500] | 0.000 [0.000–0.000] | 1.000 [0.611–1.000] | 0.000 [0.000–0.000] | 0.325 [0.030–0.641] | 0.338 [0.146–0.670] |
| **Random Forest** | 0.15 | 0.308 | 0.400 [0.206–0.500] | 0.800 [0.000–0.000] | 0.000 [0.413–1.000] | 0.471 [0.000–0.000] | 0.225 [0.000–0.515] | 0.307 [0.137–0.595] |

---

## 6. Multimodal Fusion: Adding Pre-Recording Clinical Metadata

We expanded the feature set from 57 EHG features to **61 multimodal features** by appending 4 documented clinical variables available at or before recording:
- **Maternal Age** (years)
- **Maternal Weight** (kg)
- **Parity** (prior births)
- **Recording Gestational Age (`Rectime`)** (weeks)

Missing clinical values (Age: 15.7%, Weight: 11.3%, Parity: 28.0%) were imputed using `SimpleImputer(strategy='median')` strictly inside each training fold, scaled, oversampled via SMOTE, and evaluated via 5-fold Stratified Patient CV with Youden's J threshold tuning.

### Side-by-Side Comparison: EHG-Only vs. Multimodal Fusion (TPEHGDB Cohort)

| Model | Feature Space | Balanced Accuracy (95% CI) | Sensitivity / Recall (95% CI) | Specificity (95% CI) | F1-Score (95% CI) | ROC-AUC (95% CI) | PR-AUC (95% CI) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **EHG-Only (57 feats)** | **0.539 [0.454–0.624]** | **0.368 [0.210–0.533]** | 0.710 [0.653–0.764] | **0.219 [0.126–0.323]** | **0.582 [0.489–0.671]** | **0.194 [0.118–0.306]** |
| **Logistic Regression** | **Multimodal (61 feats)** | **0.469 [0.372–0.516]** | **0.289 [0.056–0.318]** | 0.649 [0.643–0.755] | **0.156 [0.034–0.191]** | **0.488 [0.398–0.573]** | **0.153 [0.096–0.260]** |
| **SVM (RBF Kernel)** | **EHG-Only (57 feats)** | 0.471 [0.442–0.548] | 0.079 [0.024–0.219] | 0.863 [0.834–0.911] | 0.078 [0.026–0.209] | 0.476 [0.387–0.575] | 0.121 [0.082–0.186] |
| **SVM (RBF Kernel)** | **Multimodal (61 feats)** | 0.452 [0.448–0.509] | 0.053 [0.000–0.088] | 0.851 [0.888–0.950] | 0.051 [0.000–0.109] | 0.469 [0.384–0.567] | 0.117 [0.082–0.174] |
| **Random Forest** | **EHG-Only (57 feats)** | 0.459 [0.414–0.527] | 0.395 [0.023–0.219] | 0.523 [0.780–0.868] | 0.169 [0.022–0.182] | 0.430 [0.336–0.529] | 0.108 [0.075–0.159] |
| **Random Forest** | **Multimodal (61 feats)** | 0.448 [0.420–0.505] | 0.500 [0.000–0.143] | 0.397 [0.818–0.896] | 0.177 [0.000–0.137] | 0.406 [0.304–0.502] | 0.106 [0.074–0.159] |

### Honest Finding on Multimodal Fusion
Multimodal fusion with maternal age, weight, parity, and recording gestation **did not improve performance** over the EHG-only baseline (Logistic Regression Balanced Accuracy dropped from **0.539 to 0.469**, and ROC-AUC dropped from **0.582 to 0.488**).

#### Biological & Statistical Rationale:
1. **High Missingness in TPEHGDB:** Parity is missing in 28.0% of records, Age in 15.7%, and Weight in 11.3%. Median imputation flattens variance and injects uninformative centroid noise into the linear classifier.
2. **Weak Unconditional Marginal Predictors:** In a general unselected obstetrical population, age, weight, and parity without definitive clinical risk drivers (such as transvaginal sonographic cervical length $< 25\text{ mm}$ or prior spontaneous preterm birth history, neither of which was recorded in TPEHGDB) do not provide clear separability.
3. **Balanced Recording Timing:** Recording gestational age (`Rectime`) is distributed symmetrically across preterm and term cohorts by study design (both groups were sampled at early $<26\text{ wk}$ and late $\ge 26\text{ wk}$ windows), rendering `Rectime` uninformative for class discrimination while adding parameter dimensionality.

---

---

## 8. Reproducibility Standard, Fixed Seed (random_state=42), & 10-Seed Robustness Check

### Fixed Seed Standard
To ensure absolute reproducibility across all execution environments, the random seed is fixed and hard-coded to **`random_state=42`** throughout the codebase:
- **`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`**: Deterministic patient-level fold generation.
- **Model Initializations (`LogisticRegression`, `RandomForestClassifier`, `SVC`, `XGBClassifier`)**: Deterministic solver and tree bagging.
- **Bootstrap Resampling (`compute_bootstrap_ci`)**: Deterministic percentile bootstrap using `np.random.default_rng(42)`.
- **SMOTE (`SMOTE(random_state=42)`)**: Deterministic synthetic sample interpolation.
- **PyTorch (`torch.manual_seed(42)`)**: Deterministic neural network weight initialization.

### Root Cause of Baseline Score Variation (0.582 vs 0.487)
In earlier runs, the unmodified Logistic Regression baseline reported two different ROC-AUC values (0.582 vs 0.487):
1. **Unstratified Group Split (AUC = 0.582, Bal Acc = 0.539):** In the initial pipeline, `(df['label'] == 'Preterm')` was evaluated on an already-integer column (`0` and `1`), resulting in all zeros passed to `StratifiedGroupKFold`. The splitter fell back to standard unstratified `GroupKFold`, where preterm cases clustered unevenly into folds.
2. **Proper Stratified Group Split (AUC = 0.487, Bal Acc = 0.452):** When binary integer labels are properly passed to `StratifiedGroupKFold` with seed 42, preterm cases are evenly stratified across patient groups.

### 10-Seed Variance & Robustness Benchmark (Seeds 42 to 51)
To determine true run-to-run variance, both the **Center 60s Baseline** and the **Max-Energy 60s** pipelines were evaluated across 10 distinct fixed seeds (`42, 43, 44, 45, 46, 47, 48, 49, 50, 51`) using identical 5-fold Stratified Patient CV:

| Metric | Baseline (Center 60s, Mean ± Std) | Max-Energy (Peak 60s, Mean ± Std) | Paired Seed Delta (Max - Base) |
| :--- | :---: | :---: | :---: |
| **ROC-AUC** | **0.535 ± 0.035** (range: 0.487–0.600) | **0.549 ± 0.031** (range: 0.508–0.593) | **+0.014 ± 0.040** |
| **Balanced Accuracy** | **0.497 ± 0.029** (range: 0.452–0.528) | **0.523 ± 0.022** (range: 0.481–0.550) | **+0.026 ± 0.033** |
| **Sensitivity / Recall** | **0.300 ± 0.048** (range: 0.237–0.368) | **0.392 ± 0.044** (range: 0.289–0.447) | **+0.092 ± 0.061** |
| **Specificity** | **0.694 ± 0.022** (range: 0.660–0.729) | **0.653 ± 0.020** (range: 0.622–0.679) | **-0.041 ± 0.029** |

#### Per-Seed Breakdown (10 Seeds)
| Seed | Baseline AUC | Max-Energy AUC | AUC Delta | Baseline Bal Acc | Max-Energy Bal Acc | Max-Energy Wins? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | 0.487 | 0.561 | +0.074 | 0.452 | 0.524 | Yes |
| **43** | 0.530 | 0.543 | +0.013 | 0.513 | 0.531 | Yes |
| **44** | 0.499 | 0.508 | +0.009 | 0.471 | 0.514 | Yes |
| **45** | 0.512 | 0.532 | +0.020 | 0.477 | 0.516 | Yes |
| **46** | 0.600 | 0.593 | -0.007 | 0.522 | 0.550 | No (Base higher) |
| **47** | 0.578 | 0.517 | -0.061 | 0.522 | 0.495 | No (Base higher) |
| **48** | 0.555 | 0.529 | -0.026 | 0.498 | 0.481 | No (Base higher) |
| **49** | 0.516 | 0.537 | +0.021 | 0.462 | 0.529 | Yes |
| **50** | 0.532 | 0.593 | +0.061 | 0.524 | 0.548 | Yes |
| **51** | 0.540 | 0.580 | +0.040 | 0.528 | 0.537 | Yes |

### Statistical Noise Band Verdict
- **Mean AUC Improvement:** $+0.014$ ($+1.4$ percentage points).
- **Baseline Standard Deviation ($\sigma_{\text{base}}$):** $0.035$.
- **Max-Energy Standard Deviation ($\sigma_{\text{max}}$):** $0.031$.
- **Combined Standard Deviation (Sum: $\sigma_{\text{base}} + \sigma_{\text{max}}$):** $0.066$.
- **Quadrature Combined Standard Deviation ($\sqrt{\sigma_{\text{base}}^2 + \sigma_{\text{max}}^2}$):** $0.047$.

> [!IMPORTANT]
> **VERDICT ON MAX-ENERGY IMPROVEMENT:**
> **The mean AUC improvement of +0.014 does NOT exceed the combined standard deviation of both pipelines (0.014 < 0.047 quadrature, 0.014 < 0.066 sum).**
> In fact, the +0.014 mean gain is smaller than the standard deviation of either individual model ($\sigma=0.035$ and $0.031$). The apparent single-seed gain observed on seed 42 (+0.074) was an outlier at the low extreme of the baseline distribution.
> Therefore, **the improvement falls entirely inside the run-to-run cross-validation noise band** and is **NOT** statistically distinguishable from random fold-splitting variance. Max-Energy selection should be documented as an intuitive heuristic that moderately stabilizes sensitivity (+9.2%), but **cannot be claimed as a statistically significant ranking improvement**.

---

## 9. Final Frozen Production Model: Regularized ExtraTrees (69 Features)

### Final Frozen Configuration
Following systematic benchmarking of linear, non-linear, and ensemble families, the production pipeline has been finalized and locked as **Regularized ExtraTrees**:
- **Algorithm:** Extremely Randomized Trees (`sklearn.ensemble.ExtraTreesClassifier`)
- **Trees & Depth:** `n_estimators = 100`, `max_depth = 3`, `min_samples_leaf = 5`
  - *Constraining depth to 3 prevents leaf overfitting on abdominal background noise while capturing non-linear interactions across frequency bands and burst geometry.*
- **Class Balancing:** `class_weight = 'balanced'` (automatically balances class loss according to inverse class frequency)
- **Oversampling:** **No SMOTE** (reverted due to boundary distortion and probability miscalibration)
- **Window Policy:** **Contraction Burst Detection (Envelope Peak 60s Window)** (scans $[180\text{ s}, 1620\text{ s}]$ with a 15-second lowpass smoothed analytic signal envelope to center directly on the peak physiological contraction)
- **Feature Space (69 Features):**
  - **57 Base Physiological Features:** RMS, Peak/Median frequencies, Band powers (0.1–0.3, 0.3–1.0, 1.0–2.0, 2.0–3.0 Hz), Fast-to-slow ratio, Core-to-total ratio, Spectral Entropy, Sample Entropy ($m=2, r=0.2\times\text{std}$), and Cross-channel correlation peaks across $S_1, S_2, S_3$.
  - **9 Topological Contraction-Shape Features (Chowdhury et al.):** Area under burst envelope, Boundary perimeter / arc length, and Concave hull deficit for channels $S_1, S_2, S_3$.
  - **3 Self-Normalized Burst-to-Baseline Ratios:** $\text{RMS}_{\text{burst}} / \text{RMS}_{\text{background}}$ for channels $S_1, S_2, S_3$ to algebraically cancel out maternal BMI and skin impedance attenuation.
- **Scaling:** `StandardScaler` fitted strictly on training data
- **Random State:** **`random_state = 42`** hard-coded everywhere
- **Artifacts Locked:** [`models/best_triage_model.joblib`](models/best_triage_model.joblib), [`models/fitted_scaler.joblib`](models/fitted_scaler.joblib), and [`models/feature_names.json`](models/feature_names.json)

### Primary Cohort Performance (300 Records, 5-Fold Stratified Patient CV, Seed 42)
| Metric | Center 60s Baseline | Max-Energy Baseline | **ExtraTrees (Final Locked)** | 95% Bootstrap Confidence Interval |
| :--- | :---: | :---: | :---: | :---: |
| **ROC-AUC** | 0.487 | 0.561 | **0.645** | **[0.550–0.740]** |
| **Balanced Accuracy** | 0.452 | 0.524 | **0.599** | **[0.517–0.689]** |
| **Raw Accuracy** | 61.3% | 60.0% | **67.3%** | **[62.0%–72.7%]** |
| **Sensitivity (Preterm Recall)** | 23.7% (9/38) | 42.1% (16/38) | **50.0%** (19/38) | **[33.3%–66.7%]** |
| **Specificity (Term Recall)** | 66.8% (175/262) | 62.6% (164/262) | **69.8%** (183/262) | **[64.3%–75.3%]** |
| **PR-AUC** | 0.182 | 0.154 | **0.235** | **[0.144–0.367]** |

### 10-Seed Robustness Summary (Seeds 42 to 51)
Across 10 independent stratified patient CV splits:
- **ExtraTrees Mean ROC-AUC:** **0.649 ± 0.019** (range: 0.617 to 0.678)
- **Baseline Mean ROC-AUC:** **0.535 ± 0.035** (range: 0.487 to 0.600)
- **Mean AUC Delta:** **+0.114 (+11.4 percentage points)**
- **Paired Head-to-Head Win Rate:** **10 out of 10 seeds (100% win rate)**
- **ExtraTrees Mean Balanced Accuracy:** **60.3% ± 1.9%** (Raw Accuracy: **68.3% ± 1.6%**)
- **ExtraTrees Mean Sensitivity:** **49.5% ± 2.7%** (catches ~50% of preterm deliveries across all runs)

### External Transfer Evaluation on Independent Cohort ($N=13$)
Evaluated on the 13 non-overlapping independent pregnant records in TPEHGT:
- **Sensitivity:** **0.600 [0.231–0.882]*** (3 out of 5 preterms detected: `tpehgt_p001`, `tpehgt_p004`, `tpehgt_p005`)
- **Specificity:** **0.375 [0.137–0.694]***
- **Accuracy:** **0.462 [0.232–0.709]***
- **ROC-AUC:** **0.600 [0.268–0.932]***

*\*Exact Wilson score intervals for proportions and Hanley-McNeil SE for ROC-AUC.*

---

## 10. How to Run & Reproduce

```bash
# Activate virtual environment
cd "c:\Users\piyus\Downloads\ML 2\term-preterm-ehg-database-1.0.1\ehg_triage"

# Run ExtraTrees Pipeline & Benchmark
python src\upgrade_to_extratrees.py

# Run Real-Time 1-Minute Inference Engine Latency Benchmark (482 ms mean latency)
python src\inference.py

# Launch Interactive Streamlit Interface
.venv\Scripts\streamlit.exe run app.py
```



