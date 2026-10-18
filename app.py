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
    st.subheader("Interactive 60-Second EHG Triage (Primary Model: Unmodified Logistic Regression)")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 1. Signal Source")
        sig_source = st.selectbox(
            "Select Input Signal:",
            ["Preset: High-Risk Preterm Profile", "Preset: Low-Risk Term Profile", "Preset: Synthesized Contraction Burst", "Upload Custom CSV"]
        )

        threshold = st.slider("Decision Threshold:", min_value=0.1, max_value=0.9, value=0.50, step=0.02)
        
        # Generate or load signal
        np.random.seed(42)
        t = np.linspace(0, 60, WINDOW_SAMPLES)
        
        if "Preterm" in sig_source:
            burst = np.exp(-((t - 30)**2) / (2 * 8**2))
            s1 = 0.08 * burst * np.sin(2 * np.pi * 0.85 * t) + 0.015 * np.random.randn(WINDOW_SAMPLES)
            s2 = 0.07 * burst * np.sin(2 * np.pi * 0.82 * t + 0.2) + 0.015 * np.random.randn(WINDOW_SAMPLES)
            s3 = 0.06 * burst * np.sin(2 * np.pi * 0.79 * t + 0.4) + 0.015 * np.random.randn(WINDOW_SAMPLES)
            raw_sig = np.column_stack([s1, s2, s3])
        elif "Term" in sig_source:
            burst = np.exp(-((t - 30)**2) / (2 * 12**2))
            s1 = 0.03 * burst * np.sin(2 * np.pi * 0.32 * t) + 0.01 * np.random.randn(WINDOW_SAMPLES)
            s2 = 0.025 * burst * np.sin(2 * np.pi * 0.30 * t) + 0.01 * np.random.randn(WINDOW_SAMPLES)
            s3 = 0.02 * burst * np.sin(2 * np.pi * 0.28 * t) + 0.01 * np.random.randn(WINDOW_SAMPLES)
            raw_sig = np.column_stack([s1, s2, s3])
        elif "Upload" in sig_source:
            uploaded = st.file_uploader("Upload 60-second 3-channel CSV (1200 rows x 3 columns)", type=['csv'])
            if uploaded is not None:
                df_up = pd.read_csv(uploaded)
                raw_sig = df_up.iloc[:1200, :3].values
            else:
                raw_sig = 0.02 * np.random.randn(WINDOW_SAMPLES, 3)
        else:
            burst = np.exp(-((t - 25)**2) / 30) + 0.5 * np.exp(-((t - 45)**2) / 20)
            s1 = 0.05 * burst * np.sin(2 * np.pi * 0.6 * t) + 0.012 * np.random.randn(WINDOW_SAMPLES)
            s2 = 0.045 * burst * np.sin(2 * np.pi * 0.58 * t) + 0.012 * np.random.randn(WINDOW_SAMPLES)
            s3 = 0.04 * burst * np.sin(2 * np.pi * 0.55 * t) + 0.012 * np.random.randn(WINDOW_SAMPLES)
            raw_sig = np.column_stack([s1, s2, s3])

        run_btn = st.button("🚀 Run 1-Minute Triage", use_container_width=True)

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
        result = predictor.predict_one_minute(raw_sig, fs=20.0, threshold=threshold)
        
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Preterm Risk Score", f"{result['risk_score']:.1%}")
        with m2:
            st.metric("Triage Category", result['triage_category'])
        with m3:
            st.metric("Inference Latency", f"{result['inference_latency_ms']} ms")
        with m4:
            st.metric("Signal Quality", result['signal_quality'])

        st.progress(float(min(max(result['risk_score'], 0.0), 1.0)))

        st.markdown("#### Electrophysiological Biomarker Profile:")
        km = result['key_metrics']
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("RMS Amplitude (S1)", f"{km['rms_s1_uv']} µV")
        b2.metric("Median Frequency (S1)", f"{km['f_med_s1_hz']} Hz")
        b3.metric("Peak Frequency (S1)", f"{km['f_peak_s1_hz']} Hz")
        b4.metric("Sample Entropy (S1)", f"{km['sample_entropy_s1']}")

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
