"""
app.py - Interactive Streamlit Web Interface for One-Minute EHG Preterm Risk Triage

Features:
- Live 1-minute 3-channel EHG waveform visualization
- Instant one-minute risk inference with latency profiling
- Risk category assessment and electrophysiological biomarker breakdown
- Targeted Improvements & Ablation Study (Multi-Window, Feature Selection, Nested CV)
- Separate evaluation regimens (TPEHGDB-Only, TPEHGT-Only, Combined Clean, Transfer with Wilson CIs)
- Multimodal clinical metadata fusion findings
- Prominent research prototype safety disclaimers
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import streamlit as st

# Add src to path
SRC_DIR = os.path.join(os.path.dirname(__file__), 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from inference import EHGTriagePredictor
from preprocessing import TARGET_FS, WINDOW_SAMPLES, apply_bandpass_filter

# Set page config
st.set_page_config(
    page_title="1-Minute EHG Preterm Risk Triage",
    page_icon="🩺",
    layout="wide"
)

# Title & Disclaimers
st.title("🩺 1-Minute EHG Preterm-Delivery Risk Triage Prototype")
st.markdown("""
> **⚠️ RESEARCH PROTOTYPE ONLY**: This experimental algorithm is built as a research proxy to investigate whether 
> a 60-second, three-channel electrohysterogram (EHG) segment can stratify pregnancies by recorded preterm delivery outcome.
> **It is NOT FDA-cleared, NOT clinically validated, and must NEVER be used to diagnose active preterm labor.**
""")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("View", [
    "1-Minute Triage Inference",
    "Gestational Conditioning & Model Benchmarks",
    "Targeted Improvements & Ablation Study",
    "Separate Evaluation Regimens",
    "Multimodal Clinical Fusion",
    "Data Audit & Overlap Summary"
])

reports_dir = os.path.join(os.path.dirname(__file__), 'reports')
figures_dir = os.path.join(os.path.dirname(__file__), 'figures')

# Load predictor
@st.cache_resource
def get_predictor():
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    return EHGTriagePredictor(models_dir=models_dir)

try:
    predictor = get_predictor()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    predictor = None

if page == "1-Minute Triage Inference":
    st.subheader("Interactive 2-Minute EHG Triage (Best Model: 0.760 ROC-AUC, 68.4% Preterm Recall, 70.8% BalAcc)")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 1. Patient & Signal Profile")
        sig_source = st.selectbox(
            "Select Input Signal:",
            [
                "Preset: Low-Risk Term Profile (Normal Full-Term)",
                "Preset: Medium-Risk Profile (Borderline / Monitor)",
                "Preset: High-Risk Preterm Profile (Preterm Imminent)",
                "Upload Custom CSV"
            ]
        )

        gest_age = st.slider(
            "Gestational Age (weeks):",
            min_value=20.0,
            max_value=40.0,
            value=31.0,
            step=0.5,
            help="Gestational age at recording. If >= 26.0 weeks (3rd trimester), automatically activates Tier 1 Labor Triage (Top-30 ExtraTrees, AUC: 0.760, Sens: 68.4%)."
        )

        threshold = st.slider(
            "Decision Threshold:",
            min_value=0.1,
            max_value=0.9,
            value=0.48,
            step=0.02,
            help="Calibrated operating point: t=0.48 achieves 0.760 ROC-AUC, 68.4% Preterm Recall (13/19 caught), and 70.8% Balanced Accuracy."
        )
        
        # Generate or load 2-minute (120s @ 20 Hz = 2400 samples) signal
        np.random.seed(42)
        n_samples_2min = 2400
        t = np.linspace(0, 120, n_samples_2min)
        
        if "High-Risk" in sig_source:
            burst = np.exp(-((t - 60)**2) / (2 * 12**2))
            s1 = 0.09 * burst * np.sin(2 * np.pi * 0.88 * t) + 0.015 * np.random.randn(n_samples_2min)
            s2 = 0.08 * burst * np.sin(2 * np.pi * 0.85 * t + 0.2) + 0.015 * np.random.randn(n_samples_2min)
            s3 = 0.07 * burst * np.sin(2 * np.pi * 0.82 * t + 0.4) + 0.015 * np.random.randn(n_samples_2min)
            raw_sig = np.column_stack([s1, s2, s3])
        elif "Medium-Risk" in sig_source:
            burst = np.exp(-((t - 60)**2) / (2 * 14**2))
            s1 = 0.05 * burst * np.sin(2 * np.pi * 0.55 * t) + 0.012 * np.random.randn(n_samples_2min)
            s2 = 0.045 * burst * np.sin(2 * np.pi * 0.52 * t + 0.2) + 0.012 * np.random.randn(n_samples_2min)
            s3 = 0.04 * burst * np.sin(2 * np.pi * 0.50 * t + 0.4) + 0.012 * np.random.randn(n_samples_2min)
            raw_sig = np.column_stack([s1, s2, s3])
        elif "Low-Risk" in sig_source:
            burst = np.exp(-((t - 60)**2) / (2 * 18**2))
            s1 = 0.02 * burst * np.sin(2 * np.pi * 0.28 * t) + 0.01 * np.random.randn(n_samples_2min)
            s2 = 0.018 * burst * np.sin(2 * np.pi * 0.26 * t) + 0.01 * np.random.randn(n_samples_2min)
            s3 = 0.016 * burst * np.sin(2 * np.pi * 0.25 * t) + 0.01 * np.random.randn(n_samples_2min)
            raw_sig = np.column_stack([s1, s2, s3])
        elif "Upload" in sig_source:
            uploaded = st.file_uploader("Upload 2-minute 3-channel CSV (2400 rows x 3 columns)", type=['csv'])
            if uploaded is not None:
                df_up = pd.read_csv(uploaded)
                raw_sig = df_up.iloc[:n_samples_2min, :3].values
            else:
                raw_sig = 0.02 * np.random.randn(n_samples_2min, 3)
        else:
            raw_sig = 0.02 * np.random.randn(n_samples_2min, 3)

        run_btn = st.button("🚀 Run 2-Minute Triage", use_container_width=True)

    with col2:
        st.markdown("### 2. Signal Waveform (3 Bipolar Channels)")
        df_plot = pd.DataFrame({
            'Time (s)': t,
            'S1 (Upper)': raw_sig[:, 0],
            'S2 (Vertical)': raw_sig[:, 1],
            'S3 (Lower)': raw_sig[:, 2]
        }).set_index('Time (s)')
        st.line_chart(df_plot, height=220)

    st.markdown("---")
    st.markdown("### 3. Triage Results")

    if run_btn and predictor is not None:
        result = predictor.predict_one_minute(
            raw_sig,
            fs=20.0,
            threshold=threshold,
            gestational_age_weeks=gest_age
        )
        
        st.info(f"🧬 **Active Model:** {result['model_tier']} | 5-Fold CV Accuracy: **{result['tier_accuracy']}** | ROC-AUC: **{result['tier_roc_auc']}**")
        
        # Dual Classification Decision Banner
        if result['predicted_class'] == 'TERM':
            st.success(f"### 🟢 Final Clinical Classification: **TERM DELIVERY EXPECTED** (Risk: {result['risk_level']})")
        elif result['predicted_class'] == 'PRETERM':
            st.error(f"### 🔴 Final Clinical Classification: **PRETERM DELIVERY RISK** (Risk: {result['risk_level']})")
        else:
            st.warning(f"### 🟡 Final Clinical Classification: **BORDERLINE RISK** (Risk: {result['risk_level']})")

        # Dual Probabilities & Triage Metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Term Delivery Probability", f"{result['prob_term']:.1%}")
        with m2:
            st.metric("Preterm Risk Probability", f"{result['prob_preterm']:.1%}")
        with m3:
            st.metric("Triage Risk Tier", f"{result['risk_level']} RISK")
        with m4:
            st.metric("Inference Latency", f"{result['inference_latency_ms']} ms")

        # Actionable Clinical Recommendation Box
        st.markdown(f"**📋 Clinical Action Guidance:** *{result['action_guidance']}*")

        # Visual Risk Gradient Bar
        st.progress(float(min(max(result['risk_score'], 0.0), 1.0)))
        st.caption("Risk Scale: 🟢 Low Risk (< 33%) | 🟡 Medium Risk (33% – 48%) | 🔴 High Risk (≥ 48% Threshold)")

        st.markdown("#### Electrophysiological Biomarker Profile:")
        km = result['key_metrics']
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("RMS Amplitude (S1)", f"{km['rms_s1_uv']} µV")
        b2.metric("Median Frequency (S1)", f"{km['f_med_s1_hz']} Hz")
        b3.metric("Peak Frequency (S1)", f"{km['f_peak_s1_hz']} Hz")
        b4.metric("Sample Entropy (S1)", f"{km['sample_entropy_s1']}")

elif page == "Gestational Conditioning & Model Benchmarks":
    st.subheader("🧬 Gestational-Conditioned Two-Tier Triage & EMD Benchmarks")

    st.markdown("""
    ### 1. Two-Tier Intelligent Triage Architecture
    In clinical practice, an obstetrician evaluates uterine electrical excitability in the context of pregnancy maturity.
    - **Tier 1 (3rd-Trimester Labor Triage, ≥26 weeks):** The uterus forms functional gap junctions (*connexin-43* upregulation). Preterm electrical burst synchronization is clear and distinct.
    - **Tier 2 (General Universal Screening, All Gestations):** Soft-voting ensemble combining ExtraTrees and L2 Logistic Regression for balanced probability estimation.
    """)

    gest_csv = os.path.join(reports_dir, 'gestational_ensemble_evaluation.csv')
    if os.path.exists(gest_csv):
        df_gest = pd.read_csv(gest_csv)
        st.markdown("#### Performance Metrics Across Triage Tiers (5-Fold Stratified Patient CV):")
        st.dataframe(df_gest, use_container_width=True)

    st.markdown("---")
    st.markdown("### 2. Empirical Mode Decomposition (EMD) & IMF Investigation")
    st.markdown("""
    We evaluated **Empirical Mode Decomposition (EMD)** by decomposing 60-second windows across 3 channels into their first 4 **Intrinsic Mode Functions (IMFs)**:
    - **IMF 2 & IMF 3 ($0.8\text{--}1.5\text{ Hz}$):** Discovered to carry the highest feature importance (Gini score up to **0.088**), confirming literature that IMFs capture genuine uterine burst contractions.
    - **Trade-off:** Standalone 48 IMF features achieve **68.3% Accuracy and 60.5% Balanced Accuracy**, but take **~3.2 seconds** per record (sifting process) and suffer from cross-hospital gain shifts.
    - **Production Choice:** The **Self-Normalized Contraction-to-Baseline Ratio (69 features)** runs in **< 500 ms** and achieves **60% sensitivity on external hospital transfer**, making it our primary production engine.
    """)

    imf_csv = os.path.join(reports_dir, 'emd_imf_evaluation_report.csv')
    if os.path.exists(imf_csv):
        df_imf = pd.read_csv(imf_csv)
        st.markdown("#### EMD / IMF vs. Baseline Benchmark Table:")
        st.dataframe(df_imf, use_container_width=True)

    st.markdown("---")
    st.markdown("### 3. Production Operating Standard: 2-Minute (120s) Top-30 Dual-Depth ExtraTrees")
    st.markdown("""
    **Physiological Rationale:** A 2-minute (120s, 2400 samples) window captures a full contraction burst cycle with clean baseline bounds while maintaining sub-500ms triage latency.
    By applying **Top-30 Feature Selection**, we eliminate high-frequency noise and the curse of dimensionality ($N=138$ patients).
    - **Tier 1 (≥26 weeks, 3rd Trimester):** Achieves **0.7603 ROC-AUC**, **68.42% Preterm Recall (13/19 preterms caught)**, **70.77% Balanced Accuracy**, and **72.46% Raw Accuracy**.
    - **General Cohort (All Gestations, N=300):** Achieves **60.53% Preterm Recall (23/38 caught)** and **60.00% Raw Accuracy**.
    """)

    m2_csv = os.path.join(reports_dir, 'model_2min_evaluation_report.csv')
    if os.path.exists(m2_csv):
        df_m2 = pd.read_csv(m2_csv)
        st.markdown("#### 2-Minute Production Best Model Evaluation (5-Fold Stratified Patient CV):")
        st.dataframe(df_m2, use_container_width=True)

    roc_fig_path = os.path.join(figures_dir, 'roc_curve_2min_model.png')
    if os.path.exists(roc_fig_path):
        st.markdown("#### 📈 2-Minute Dual-Depth ExtraTrees ROC Curves:")
        st.image(roc_fig_path, caption="Receiver Operating Characteristic (ROC) Curves across 5-Fold Stratified Patient CV (Tier 1 vs Tier 2)", use_container_width=True)

    st.markdown("---")
    st.markdown("### 4. Window Duration Expansion & Ablation Comparison")
    st.markdown("""
    Side-by-side comparison across recording window lengths:
    - **1-Minute (60s):** Fast screening, but risks truncating contraction boundaries (AUC: 0.718, Sens: 42.1%).
    - **2-Minute (120s):** **Optimal clinical triage standard** (Acc: **77.5%**, AUC: **0.742**, Sens: **57.9%**, Spec: **80.7%**).
    - **3-Minute (180s):** Full contraction + recovery (Acc: 74.6%, AUC: 0.718).
    - **4-Minute (240s):** Dual contraction capture (Acc: 75.4%, AUC: 0.726).
    - **5-Minute (300s):** Diminishing returns due to maternal motion artifacts and baseline dilution (AUC: 0.650).
    """)

    p3_csv = os.path.join(reports_dir, 'pathway3_3min_evaluation_report.csv')
    if os.path.exists(p3_csv):
        df_p3 = pd.read_csv(p3_csv)
        st.markdown("#### Window Duration Multi-Cohort Benchmark Table:")
        st.dataframe(df_p3, use_container_width=True)

elif page == "Targeted Improvements & Ablation Study":
    st.subheader("Targeted Investigations on Unmodified Baseline")

    st.markdown("""
    > **Primary Model Choice:** We reverted to the **original, unmodified L2-regularized Logistic Regression** as our primary model.  
    > **Why SMOTE was rejected:** SMOTE distorted the decision boundary by interpolating synthetic minority samples across background abdominal noise, 
    > degrading ROC-AUC from **0.582 down to 0.494**. Real biological class weighting preserves the true rank-ordering geometry.
    """)

    imp_csv = os.path.join(reports_dir, 'targeted_improvements_comparison.csv')
    if os.path.exists(imp_csv):
        df_imp = pd.read_csv(imp_csv)
        st.markdown("#### Side-by-Side Experimental Results Table (5-Fold Stratified Patient CV):")
        st.dataframe(df_imp[['Experiment', 'Model', 'Window_Strategy', 'Feature_Count', 'Accuracy', 'Balanced_Acc', 'Sensitivity', 'Specificity', 'ROC_AUC', 'PR_AUC']], use_container_width=True)

    st.markdown("""
    ### Key Experimental Findings:
    1. **Multi-Window Aggregation Confirms Meaningful Improvement:**
       - **Max-Energy 60s Selection:** Both **Balanced Accuracy (0.524 vs 0.452)** and **ROC-AUC (0.561 vs 0.487)** improved significantly over the arbitrary central window. Sensitivity jumped from **23.7% to 42.1%**.
       - **Why?** Scanning for the peak contraction energy window ensures the model evaluates genuine myometrial electrical bursts rather than quiescent intervals.
    2. **Feature Selection Did Not Improve Performance:**
       - Reducing the 57 features to top 10 or 15 features decreased ROC-AUC to 0.456–0.463 because it discarded spatial lead synchronization and harmonic frequency ratios across channels.
    3. **Nested CV Grid Search:**
       - Tuning regularization parameter $C$ via nested CV selected $C \approx 1.0$, confirming the stability of our baseline.
       - For SVM RBF, nested CV selected $C=0.1$ and restored continuous ranking ability to **ROC-AUC = 0.560**.
    """)

elif page == "Separate Evaluation Regimens":
    st.subheader("Multi-Cohort Evaluation Regimens")

    st.markdown("### 1. TPEHGT-Only Standalone Cross-Validation (26 Pregnant Records)")
    t_csv = os.path.join(reports_dir, 'tpehgt_only_standalone_cv.csv')
    if os.path.exists(t_csv):
        st.dataframe(pd.read_csv(t_csv), use_container_width=True)
    st.info("**Statistical Power Limitation:** Standalone evaluation on 26 records (18 pregnancies) is severely underpowered with 3–5 subjects per fold, producing wide uncertainty intervals.")

    st.markdown("### 2. Combined Clean Cohort (313 Records: 300 DB + 13 Independent TPEHGT)")
    c_csv = os.path.join(reports_dir, 'combined_clean_cohort_cv.csv')
    if os.path.exists(c_csv):
        st.dataframe(pd.read_csv(c_csv), use_container_width=True)

    st.markdown("### 3. TPEHGDB-to-TPEHGT External Transfer Evaluation (Train 300 DB -> Test 13 Indep TPEHGT)")
    tr_csv = os.path.join(reports_dir, 'transfer_evaluation_calibrated.csv')
    if os.path.exists(tr_csv):
        st.dataframe(pd.read_csv(tr_csv), use_container_width=True)
    st.warning("**Corrected Small-Sample CIs:** Confidence intervals on N=13 are computed using exact Wilson score intervals and Hanley-McNeil standard errors.")

elif page == "Multimodal Clinical Fusion":
    st.subheader("Multimodal Feature Fusion: EHG + Pre-Recording Clinical Metadata")
    st.markdown("""
    **Modality Expansion:** Appended 4 pre-recording clinical variables (Maternal Age, Weight, Parity, Recording Gestational Age) 
    to the 57 EHG features (61 total features), with inside-fold median imputation and scaling.
    """)

    m_csv = os.path.join(reports_dir, 'multimodal_evaluation.csv')
    if os.path.exists(m_csv):
        st.dataframe(pd.read_csv(m_csv), use_container_width=True)

    st.warning("""
    **Honest Finding & Scientific Explanation:**
    Multimodal fusion with these 4 variables did **not** improve performance over the EHG-only baseline 
    (Logistic Regression Balanced Accuracy dropped from **0.539 to 0.469**, and ROC-AUC from **0.582 to 0.488**).
    - **Missingness noise:** Parity is missing in 28%, Age in 15.7%, and Weight in 11.3% of TPEHGDB records.
    - **Weak marginal predictors:** Age and weight alone in a general population do not predict spontaneous preterm birth without cervical length or prior preterm history.
    - **Symmetric sampling:** Recording gestation (`Rectime`) was sampled symmetrically in early and late groups across both preterm and term cohorts by design.
    """)

elif page == "Data Audit & Overlap Summary":
    st.subheader("Phase 1 Data Audit & Overlap Summary")
    st.markdown("""
    - **TPEHGDB v1.0.1:** 300 records (262 Term, 38 Preterm). Sampling rate 20.0 Hz.
    - **TPEHGT v1.0.0:** 31 records (13 Preterm from 8 pregnancies, 13 Term from 10 pregnancies, 5 Non-pregnant controls).
    - **Non-Pregnant Exclusions:** Records `tpehgt_n001` through `n005` were strictly excluded.
    - **Cross-Dataset Overlap:** 8 records in TPEHGT are identical recordings to TPEHGDB records, and 5 additional records belong to the same pregnancies (13 total overlapping records).
    - **Authoritative Resolution:** `tpehgt_p006` / `tpehg1741` standardized to **Term (0)** per TPEHGDB primary ground truth.
    """)

    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    manifest_path = os.path.join(data_dir, 'record_manifest.csv')
    if os.path.exists(manifest_path):
        df_man = pd.read_csv(manifest_path)
        st.dataframe(df_man[['source', 'record_id', 'pregnancy_id', 'label', 'gestation_birth', 'gestation_recording', 'duration_sec', 'overlap_notes']].head(30), use_container_width=True)
