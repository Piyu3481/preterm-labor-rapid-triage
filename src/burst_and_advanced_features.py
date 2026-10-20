"""
burst_and_advanced_features.py - Systematic Evaluation of 4 Targeted Enhancements:

1. Burst-detection-based 60s window selection (analytic signal envelope peak)
2. Topological contraction-shape features (Chowdhury et al.: envelope area, perimeter, concave deficit)
3. Additional entropy features (Approximate Entropy, Shannon Entropy)
4. Calibrated soft-voting ensemble (Platt-scaled LR, SVM, Random Forest)
5. 10-seed robustness check (seeds 42-51) and 13-record external transfer validation
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.spatial import ConvexHull
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    balanced_accuracy_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    f1_score,
    average_precision_score
)

from preprocessing import load_raw_ehg_record, apply_bandpass_filter, check_signal_quality
from features import extract_features_from_window
from evaluate import compute_metrics, compute_bootstrap_ci
from targeted_experiments import wilson_score_interval


# -------------------------------------------------------------------------
# 1. Burst Detection & Window Extraction
# -------------------------------------------------------------------------

def detect_burst_60s_window(
    sig: np.ndarray,
    fs: float = 20.0,
    lowcut: float = 0.3,
    highcut: float = 3.0,
    window_sec: float = 60.0
) -> dict:
    """
    Detect the strongest contraction burst across the 30-min recording
    in the valid non-transient region [180s, 1620s] using analytic signal envelope.
    Returns 60s window centered on the burst peak.
    """
    filtered = apply_bandpass_filter(sig, lowcut=lowcut, highcut=highcut, fs=fs)
    n_samples = filtered.shape[0]
    
    # 15s low-pass filter to smooth instantaneous envelope into slow uterine contraction profile
    b, a = signal.butter(2, 0.05 / (fs / 2), btype='low')
    
    envelopes = []
    for ch in range(3):
        analytic = signal.hilbert(filtered[:, ch])
        env = np.abs(analytic)
        env_smooth = signal.filtfilt(b, a, env)
        envelopes.append(env_smooth)
        
    composite_env = np.mean(envelopes, axis=0)
    
    # Search within non-transient region [180s, 1620s]
    transient_samples = int(180 * fs)
    win_samples = int(window_sec * fs)
    half_win = win_samples // 2
    
    search_start = transient_samples + half_win
    search_end = n_samples - transient_samples - half_win
    
    if search_end <= search_start:
        # Fallback if signal is shorter than 360s + 60s
        search_start = half_win
        search_end = max(half_win + 1, n_samples - half_win)
        
    peak_idx = search_start + np.argmax(composite_env[search_start:search_end])
    
    start_idx = peak_idx - half_win
    end_idx = start_idx + win_samples
    
    window_sig = filtered[start_idx:end_idx, :]
    is_valid, reason = check_signal_quality(window_sig)
    
    return {
        'signal': window_sig,
        'start_time_sec': start_idx / fs,
        'peak_time_sec': peak_idx / fs,
        'peak_envelope_amp': float(composite_env[peak_idx]),
        'is_valid': is_valid,
        'quality_reason': reason
    }


# -------------------------------------------------------------------------
# 2. Topological & Advanced Entropy Feature Extraction
# -------------------------------------------------------------------------

def compute_approximate_entropy(x: np.ndarray, m: int = 2, r: float = None) -> float:
    """Approximate Entropy (ApEn) with m=2, r=0.2*std using accelerated antropy."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    if len(x) > 600:
        x = np.ascontiguousarray(x[::2])
    N = len(x)
    std_val = np.std(x)
    if std_val < 1e-8 or N <= m + 1:
        return 0.0

    try:
        import antropy as ant
        ae = float(ant.app_entropy(x, order=m, metric='chebyshev'))
        if np.isnan(ae) or np.isinf(ae):
            return 0.0
        return ae
    except Exception:
        pass

    if r is None:
        r = 0.2 * std_val

    def _phi(k):
        X = np.array([x[i:i + k] for i in range(N - k + 1)])
        diff = np.abs(X[:, None, :] - X[None, :, :]).max(axis=2)
        C = (diff <= r).sum(axis=1) / float(N - k + 1)
        return float(np.mean(np.log(C)))

    return float(np.abs(_phi(m) - _phi(m + 1)))


def compute_shannon_entropy(x: np.ndarray, n_bins: int = 20) -> float:
    """Shannon Entropy of amplitude distribution."""
    x = np.asarray(x, dtype=np.float64)
    if np.std(x) < 1e-8:
        return 0.0
    counts, _ = np.histogram(x, bins=n_bins, density=False)
    p = counts / float(len(x))
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def compute_topological_shape_features(win_sig: np.ndarray, fs: float = 20.0) -> dict:
    """
    Topological contraction-shape features from amplitude envelope (Chowdhury et al.):
    - Area under the burst (integral of envelope over time)
    - Perimeter / boundary length of the burst envelope curve
    - Concave hull area / deficit (non-convex area between curve and convex envelope)
    """
    feats = {}
    for ch in range(3):
        x = win_sig[:, ch]
        analytic = signal.hilbert(x)
        env = np.abs(analytic)
        b, a = signal.butter(2, 0.1 / (fs / 2), btype='low')
        env_smooth = signal.filtfilt(b, a, env)
        t = np.linspace(0, 60.0, len(env_smooth))
        
        area = float(np.trapezoid(env_smooth, t))
        arc_len = float(np.sum(np.sqrt(np.diff(t)**2 + np.diff(env_smooth)**2)))
        perimeter = float(arc_len + env_smooth[0] + env_smooth[-1] + 60.0)
        
        # 2D points of the envelope polygon bounded by x-axis
        poly_pts = np.vstack([np.column_stack([t, env_smooth]), [[t[-1], 0.0], [t[0], 0.0]]])
        hull = ConvexHull(poly_pts)
        convex_area = float(hull.volume)
        concave_deficit = float(max(0.0, convex_area - area))
        
        feats[f'ch{ch+1}_burst_area'] = area
        feats[f'ch{ch+1}_burst_perimeter'] = perimeter
        feats[f'ch{ch+1}_burst_concave_area'] = concave_deficit
        
    return feats


def extract_all_burst_features(win_sig: np.ndarray, fs: float = 20.0) -> dict:
    """Extract all 72 features (57 baseline + 9 topological + 6 entropy)."""
    # 1. Base 57 physiological features
    feats = extract_features_from_window(win_sig, fs=fs)
    
    # 2. 9 Topological shape features
    topo_feats = compute_topological_shape_features(win_sig, fs=fs)
    feats.update(topo_feats)
    
    # 3. 6 Entropy features (ApEn + Shannon)
    for ch in range(3):
        x = win_sig[:, ch]
        feats[f'ch{ch+1}_apen'] = compute_approximate_entropy(x, m=2)
        feats[f'ch{ch+1}_shannon_entropy'] = compute_shannon_entropy(x, n_bins=20)
        
    return feats


# -------------------------------------------------------------------------
# 3. Pipeline Extraction and Caching
# -------------------------------------------------------------------------

def get_or_extract_burst_features(
    tpehgdb_root: str,
    manifest_csv: str,
    output_csv: str
) -> pd.DataFrame:
    if os.path.exists(output_csv):
        print(f"Loading cached burst features from {output_csv}...")
        return pd.read_csv(output_csv)
        
    print(f"Extracting burst-detected 60s windows and all 72 features for TPEHGDB...")
    df_man = pd.read_csv(manifest_csv)
    df_db = df_man[df_man['source'] == 'TPEHGDB'].reset_index(drop=True)
    
    rows = []
    t0 = time.time()
    for idx, r in df_db.iterrows():
        rec_id = r['record_id']
        rec_path = os.path.join(tpehgdb_root, 'tpehgdb', rec_id)
        sig, fs, _ = load_raw_ehg_record(rec_path, 'TPEHGDB')
        
        burst_info = detect_burst_60s_window(sig, fs=fs)
        win = burst_info['signal']
        
        row_feats = extract_all_burst_features(win, fs=fs)
        row_feats['record_id'] = rec_id
        row_feats['pregnancy_id'] = r['pregnancy_id']
        row_feats['label'] = 1 if r['label'] == 'Preterm' else 0
        row_feats['start_time_sec'] = burst_info['start_time_sec']
        row_feats['burst_peak_time_sec'] = burst_info['peak_time_sec']
        rows.append(row_feats)
        
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/300 records ({time.time() - t0:.1f}s)...")
            
    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_csv, index=False)
    print(f"Cached all {len(df_out)} burst feature records to {output_csv}")
    return df_out


def get_or_extract_tpehgt_burst_features(
    tpehgt_root: str,
    manifest_csv: str,
    output_csv: str
) -> pd.DataFrame:
    if os.path.exists(output_csv):
        print(f"Loading cached TPEHGT burst features from {output_csv}...")
        return pd.read_csv(output_csv)
        
    print("Extracting burst-detected 60s windows for independent TPEHGT cohort...")
    df_man = pd.read_csv(manifest_csv)
    independent_tpehgt = [
        'tpehgt_p001', 'tpehgt_p004', 'tpehgt_p005', 'tpehgt_p012', 'tpehgt_p013',
        'tpehgt_t001', 'tpehgt_t002', 'tpehgt_t003', 'tpehgt_t004', 'tpehgt_t005',
        'tpehgt_t006', 'tpehgt_t010', 'tpehgt_t013'
    ]
    
    rows = []
    for rec_id in independent_tpehgt:
        rec_path = os.path.join(tpehgt_root, rec_id)
        sig, fs, _ = load_raw_ehg_record(rec_path, 'TPEHGT')
        burst_info = detect_burst_60s_window(sig, fs=fs)
        win = burst_info['signal']
        
        row_feats = extract_all_burst_features(win, fs=fs)
        r = df_man[df_man['record_id'] == rec_id].iloc[0]
        row_feats['record_id'] = rec_id
        row_feats['pregnancy_id'] = r['pregnancy_id']
        row_feats['label'] = 1 if r['label'] == 'Preterm' else 0
        row_feats['start_time_sec'] = burst_info['start_time_sec']
        row_feats['burst_peak_time_sec'] = burst_info['peak_time_sec']
        rows.append(row_feats)
        
    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_csv, index=False)
    print(f"Cached {len(df_out)} independent TPEHGT burst records to {output_csv}")
    return df_out


# -------------------------------------------------------------------------
# 4. Evaluation Engine (Patient-Level 5-Fold Stratified CV)
# -------------------------------------------------------------------------

def run_patient_cv_eval(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_type: str = 'logistic_regression',
    random_state: int = 42
) -> np.ndarray:
    """
    Run 5-Fold StratifiedGroupKFold CV.
    Supports 'logistic_regression' and 'calibrated_ensemble'.
    """
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    oof_probs = np.zeros(len(y))
    
    for train_idx, val_idx in sgkf.split(X, y, groups):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_train)
        X_v_sc = scaler.transform(X_val)
        
        if model_type == 'logistic_regression':
            clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=random_state)
            clf.fit(X_tr_sc, y_train)
            oof_probs[val_idx] = clf.predict_proba(X_v_sc)[:, 1]
            
        elif model_type == 'calibrated_ensemble':
            # Base models
            lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=random_state)
            svm = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', probability=True, random_state=random_state)
            rf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=random_state)
            
            # Platt scaling (sigmoid calibration) fit strictly inside training fold via 3-fold internal CV
            cal_lr = CalibratedClassifierCV(estimator=lr, method='sigmoid', cv=3)
            cal_svm = CalibratedClassifierCV(estimator=svm, method='sigmoid', cv=3)
            cal_rf = CalibratedClassifierCV(estimator=rf, method='sigmoid', cv=3)
            
            cal_lr.fit(X_tr_sc, y_train)
            cal_svm.fit(X_tr_sc, y_train)
            cal_rf.fit(X_tr_sc, y_train)
            
            p_lr = cal_lr.predict_proba(X_v_sc)[:, 1]
            p_svm = cal_svm.predict_proba(X_v_sc)[:, 1]
            p_rf = cal_rf.predict_proba(X_v_sc)[:, 1]
            
            # Soft-voting average
            p_ens = (p_lr + p_svm + p_rf) / 3.0
            oof_probs[val_idx] = p_ens
            
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
            
    return oof_probs


def evaluate_metrics_with_ci(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.50) -> dict:
    m = compute_metrics(y_true, y_prob, threshold=threshold)
    ci = compute_bootstrap_ci(y_true, y_prob, threshold=threshold, random_state=42)
    return {
        'accuracy': f"{m['accuracy']:.3f} [{ci['accuracy']['ci_lower']:.3f}-{ci['accuracy']['ci_upper']:.3f}]",
        'balanced_accuracy': f"{m['balanced_accuracy']:.3f} [{ci['balanced_accuracy']['ci_lower']:.3f}-{ci['balanced_accuracy']['ci_upper']:.3f}]",
        'sensitivity': f"{m['sensitivity_recall']:.3f} [{ci['sensitivity_recall']['ci_lower']:.3f}-{ci['sensitivity_recall']['ci_upper']:.3f}]",
        'specificity': f"{m['specificity']:.3f} [{ci['specificity']['ci_lower']:.3f}-{ci['specificity']['ci_upper']:.3f}]",
        'precision': f"{m['precision']:.3f} [{ci['precision']['ci_lower']:.3f}-{ci['precision']['ci_upper']:.3f}]",
        'f1_score': f"{m['f1_score']:.3f} [{ci['f1_score']['ci_lower']:.3f}-{ci['f1_score']['ci_upper']:.3f}]",
        'roc_auc': f"{m['roc_auc']:.3f} [{ci['roc_auc']['ci_lower']:.3f}-{ci['roc_auc']['ci_upper']:.3f}]",
        'pr_auc': f"{m['pr_auc']:.3f} [{ci['pr_auc']['ci_lower']:.3f}-{ci['pr_auc']['ci_upper']:.3f}]",
        'raw_bal_acc': m['balanced_accuracy'],
        'raw_sens': m['sensitivity_recall'],
        'raw_roc_auc': m['roc_auc']
    }


# -------------------------------------------------------------------------
# 5. Master Workflow Execution
# -------------------------------------------------------------------------

def run_all_enhancements():
    base_dir = r'c:\Users\piyus\Downloads\ML 2\term-preterm-ehg-database-1.0.1\ehg_triage'
    tpehgdb_root = r'c:\Users\piyus\Downloads\ML 2\term-preterm-ehg-database-1.0.1\term-preterm-ehg-database-1.0.1'
    tpehgt_root = r'c:\Users\piyus\Downloads\ML 2\term-preterm-ehg-dataset-with-tocogram-1.0.0\term-preterm-ehg-dataset-with-tocogram-1.0.0'
    manifest_csv = os.path.join(base_dir, 'data', 'record_manifest.csv')
    reports_dir = os.path.join(base_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    burst_csv = os.path.join(base_dir, 'data', 'features_60s_burst_detected_all.csv')
    df_burst = get_or_extract_burst_features(tpehgdb_root, manifest_csv, burst_csv)
    
    tpehgt_burst_csv = os.path.join(base_dir, 'data', 'features_60s_burst_detected_tpehgt13.csv')
    df_tpehgt_burst = get_or_extract_tpehgt_burst_features(tpehgt_root, manifest_csv, tpehgt_burst_csv)
    
    y = df_burst['label'].values
    groups = df_burst['pregnancy_id'].values
    
    # Define feature subsets
    non_feat_cols = ['record_id', 'pregnancy_id', 'label', 'start_time_sec', 'burst_peak_time_sec']
    all_cols = [c for c in df_burst.columns if c not in non_feat_cols]
    
    topo_cols = [c for c in all_cols if 'burst_area' in c or 'burst_perimeter' in c or 'burst_concave_area' in c]
    entropy_cols = [c for c in all_cols if 'apen' in c or 'shannon_entropy' in c]
    base_57_cols = [c for c in all_cols if c not in topo_cols and c not in entropy_cols]
    
    print(f"Feature schema: {len(base_57_cols)} base + {len(topo_cols)} topological + {len(entropy_cols)} entropy = {len(all_cols)} total")
    
    comparison_table_rows = []
    
    # -------------------------------------------------------------
    # Reference 0A: Validated Center 60s Baseline
    # Reference 0B: Validated Max-Energy 60s Baseline
    # -------------------------------------------------------------
    print("\n--- Evaluating Baselines for Reference ---")
    df_center = pd.read_csv(os.path.join(base_dir, 'data', 'features_60s_center.csv'))
    df_center = df_center[df_center['source'] == 'TPEHGDB'].reset_index(drop=True)
    f_c = [c for c in df_center.columns if c not in ['record_id', 'source', 'pregnancy_id', 'label', 'start_time_sec']]
    p_center = run_patient_cv_eval(df_center[f_c].values, df_center['label'].values, df_center['pregnancy_id'].values, 'logistic_regression', 42)
    m_center = evaluate_metrics_with_ci(df_center['label'].values, p_center)
    comparison_table_rows.append({
        'Step': '0A. Baseline (Center 60s)',
        'Model': 'Logistic Regression',
        'Window_Strategy': 'Center 60s Window',
        'Feature_Set': '57 Base EHG',
        'Feature_Count': 57,
        'Accuracy': m_center['accuracy'],
        'Balanced_Acc': m_center['balanced_accuracy'],
        'Sensitivity': m_center['sensitivity'],
        'Specificity': m_center['specificity'],
        'ROC_AUC': m_center['roc_auc'],
        'PR_AUC': m_center['pr_auc'],
        'Raw_Bal_Acc': m_center['raw_bal_acc'],
        'Raw_ROC_AUC': m_center['raw_roc_auc']
    })
    
    df_max = pd.read_csv(os.path.join(base_dir, 'data', 'features_60s_max_energy.csv'))
    f_m = [c for c in df_max.columns if c not in ['record_id', 'pregnancy_id', 'label', 'start_time_sec']]
    p_max = run_patient_cv_eval(df_max[f_m].values, df_max['label'].values, df_max['pregnancy_id'].values, 'logistic_regression', 42)
    m_max = evaluate_metrics_with_ci(df_max['label'].values, p_max)
    comparison_table_rows.append({
        'Step': '0B. Baseline (Max-Energy 60s)',
        'Model': 'Logistic Regression',
        'Window_Strategy': 'Max-Energy 60s Window',
        'Feature_Set': '57 Base EHG',
        'Feature_Count': 57,
        'Accuracy': m_max['accuracy'],
        'Balanced_Acc': m_max['balanced_accuracy'],
        'Sensitivity': m_max['sensitivity'],
        'Specificity': m_max['specificity'],
        'ROC_AUC': m_max['roc_auc'],
        'PR_AUC': m_max['pr_auc'],
        'Raw_Bal_Acc': m_max['raw_bal_acc'],
        'Raw_ROC_AUC': m_max['raw_roc_auc']
    })
    
    # -------------------------------------------------------------
    # Step 1: Burst Detection (57 base features)
    # -------------------------------------------------------------
    print("\n--- Step 1: Burst Detection (57 base features) ---")
    X_step1 = df_burst[base_57_cols].values
    p_step1 = run_patient_cv_eval(X_step1, y, groups, 'logistic_regression', 42)
    m_step1 = evaluate_metrics_with_ci(y, p_step1)
    comparison_table_rows.append({
        'Step': '1. Burst Detector',
        'Model': 'Logistic Regression',
        'Window_Strategy': 'Envelope Burst Peak 60s',
        'Feature_Set': '57 Base EHG',
        'Feature_Count': 57,
        'Accuracy': m_step1['accuracy'],
        'Balanced_Acc': m_step1['balanced_accuracy'],
        'Sensitivity': m_step1['sensitivity'],
        'Specificity': m_step1['specificity'],
        'ROC_AUC': m_step1['roc_auc'],
        'PR_AUC': m_step1['pr_auc'],
        'Raw_Bal_Acc': m_step1['raw_bal_acc'],
        'Raw_ROC_AUC': m_step1['raw_roc_auc']
    })
    
    # -------------------------------------------------------------
    # Step 2: Burst Detection + Topological Contraction Shape (66 features)
    # -------------------------------------------------------------
    print("\n--- Step 2: Burst Detection + Topological Contraction Shape (66 features) ---")
    step2_cols = base_57_cols + topo_cols
    X_step2 = df_burst[step2_cols].values
    p_step2 = run_patient_cv_eval(X_step2, y, groups, 'logistic_regression', 42)
    m_step2 = evaluate_metrics_with_ci(y, p_step2)
    comparison_table_rows.append({
        'Step': '2. Burst + Topological Shape',
        'Model': 'Logistic Regression',
        'Window_Strategy': 'Envelope Burst Peak 60s',
        'Feature_Set': '57 Base + 9 Topological Shape',
        'Feature_Count': len(step2_cols),
        'Accuracy': m_step2['accuracy'],
        'Balanced_Acc': m_step2['balanced_accuracy'],
        'Sensitivity': m_step2['sensitivity'],
        'Specificity': m_step2['specificity'],
        'ROC_AUC': m_step2['roc_auc'],
        'PR_AUC': m_step2['pr_auc'],
        'Raw_Bal_Acc': m_step2['raw_bal_acc'],
        'Raw_ROC_AUC': m_step2['raw_roc_auc']
    })
    
    # -------------------------------------------------------------
    # Step 3: Burst Detection + Topological + Additional Entropies (72 features)
    # -------------------------------------------------------------
    print("\n--- Step 3: Burst + Topological + Additional Entropies (72 features) ---")
    step3_cols = base_57_cols + topo_cols + entropy_cols
    X_step3 = df_burst[step3_cols].values
    p_step3 = run_patient_cv_eval(X_step3, y, groups, 'logistic_regression', 42)
    m_step3 = evaluate_metrics_with_ci(y, p_step3)
    comparison_table_rows.append({
        'Step': '3. Burst + Topo + Entropies',
        'Model': 'Logistic Regression',
        'Window_Strategy': 'Envelope Burst Peak 60s',
        'Feature_Set': '57 Base + 9 Topo + 6 Entropies',
        'Feature_Count': len(step3_cols),
        'Accuracy': m_step3['accuracy'],
        'Balanced_Acc': m_step3['balanced_accuracy'],
        'Sensitivity': m_step3['sensitivity'],
        'Specificity': m_step3['specificity'],
        'ROC_AUC': m_step3['roc_auc'],
        'PR_AUC': m_step3['pr_auc'],
        'Raw_Bal_Acc': m_step3['raw_bal_acc'],
        'Raw_ROC_AUC': m_step3['raw_roc_auc']
    })
    
    # -------------------------------------------------------------
    # Step 4: Calibrated Ensemble (Platt-scaled LR + SVM + Random Forest)
    # -------------------------------------------------------------
    print("\n--- Step 4: Calibrated Soft-Voting Ensemble ---")
    # Identify best feature set among Steps 1-3
    scores = [
        ('57_Base', step1_cols := base_57_cols, m_step1['raw_bal_acc'], m_step1['raw_roc_auc']),
        ('66_Topo', step2_cols, m_step2['raw_bal_acc'], m_step2['raw_roc_auc']),
        ('72_All', step3_cols, m_step3['raw_bal_acc'], m_step3['raw_roc_auc'])
    ]
    # Rank by combined sum (Bal Acc + ROC-AUC)
    scores.sort(key=lambda x: (x[2] + x[3]), reverse=True)
    best_feat_name, best_cols, _, _ = scores[0]
    print(f"Best feature set for ensemble: {best_feat_name} ({len(best_cols)} features)")
    
    X_step4 = df_burst[best_cols].values
    p_step4 = run_patient_cv_eval(X_step4, y, groups, 'calibrated_ensemble', 42)
    m_step4 = evaluate_metrics_with_ci(y, p_step4)
    comparison_table_rows.append({
        'Step': '4. Calibrated Ensemble',
        'Model': 'Platt Soft-Vote (LR+SVM+RF)',
        'Window_Strategy': 'Envelope Burst Peak 60s',
        'Feature_Set': f"Best ({best_feat_name})",
        'Feature_Count': len(best_cols),
        'Accuracy': m_step4['accuracy'],
        'Balanced_Acc': m_step4['balanced_accuracy'],
        'Sensitivity': m_step4['sensitivity'],
        'Specificity': m_step4['specificity'],
        'ROC_AUC': m_step4['roc_auc'],
        'PR_AUC': m_step4['pr_auc'],
        'Raw_Bal_Acc': m_step4['raw_bal_acc'],
        'Raw_ROC_AUC': m_step4['raw_roc_auc']
    })
    
    df_comp = pd.DataFrame(comparison_table_rows)
    comp_csv = os.path.join(reports_dir, 'enhancements_comparison_table.csv')
    df_comp.to_csv(comp_csv, index=False)
    print(f"\nSaved consolidated enhancements comparison to: {comp_csv}")
    
    # -------------------------------------------------------------
    # Step 5: Final Selection & Robustness Check (Seeds 42-51)
    # -------------------------------------------------------------
    print("\n=======================================================")
    print("STEP 5: Final Validation & 10-Seed Robustness Check")
    print("=======================================================")
    # Pick the best configuration across steps 1-4 on balanced accuracy and ROC-AUC jointly
    candidates = [
        ('Step 1 (Burst 57 feats, LR)', X_step1, 'logistic_regression', base_57_cols, m_step1['raw_bal_acc'], m_step1['raw_roc_auc']),
        ('Step 2 (Burst + Topo 66 feats, LR)', X_step2, 'logistic_regression', step2_cols, m_step2['raw_bal_acc'], m_step2['raw_roc_auc']),
        ('Step 3 (Burst + Topo + Entropies 72 feats, LR)', X_step3, 'logistic_regression', step3_cols, m_step3['raw_bal_acc'], m_step3['raw_roc_auc']),
        ('Step 4 (Calibrated Ensemble)', X_step4, 'calibrated_ensemble', best_cols, m_step4['raw_bal_acc'], m_step4['raw_roc_auc'])
    ]
    candidates.sort(key=lambda x: (x[4] + x[5]), reverse=True)
    best_cand_name, best_X, best_model_type, best_cand_cols, best_bal, best_auc = candidates[0]
    print(f"Top performing enhancement across Steps 1-4: {best_cand_name}")
    print(f"  Balanced Acc: {best_bal:.3f}, ROC-AUC: {best_auc:.3f}")
    
    # 10-Seed Multi-Run Variance check across seeds 42 to 51
    seeds = list(range(42, 52))
    
    # Load 10-seed results for validated baseline (Center 60s) for exact paired comparison
    base_seeds_csv = os.path.join(reports_dir, 'robustness_10seeds_log.csv')
    df_base_seeds = pd.read_csv(base_seeds_csv)
    df_base_center = df_base_seeds[df_base_seeds['strategy'] == 'Baseline_Center_60s'].sort_values('seed').reset_index(drop=True)
    
    enhancement_seed_records = []
    print(f"Running 10-seed evaluation on {best_cand_name} across seeds {seeds}...")
    for seed in seeds:
        p_seed = run_patient_cv_eval(best_X, y, groups, best_model_type, seed)
        preds_seed = (p_seed >= 0.50).astype(int)
        
        bal_acc = balanced_accuracy_score(y, preds_seed)
        sens = recall_score(y, preds_seed, pos_label=1, zero_division=0)
        spec = recall_score(y, preds_seed, pos_label=0, zero_division=0)
        roc_auc = roc_auc_score(y, p_seed)
        
        enhancement_seed_records.append({
            'seed': seed,
            'configuration': best_cand_name,
            'balanced_accuracy': bal_acc,
            'sensitivity': sens,
            'specificity': spec,
            'roc_auc': roc_auc
        })
        
    df_enh_seeds = pd.DataFrame(enhancement_seed_records)
    enh_log_csv = os.path.join(reports_dir, 'enhancements_10seeds_robustness.csv')
    df_enh_seeds.to_csv(enh_log_csv, index=False)
    
    # Paired seed comparison against baseline
    enh_auc_mean = df_enh_seeds['roc_auc'].mean()
    enh_auc_std = df_enh_seeds['roc_auc'].std(ddof=1)
    base_auc_mean = df_base_center['roc_auc'].mean()
    base_auc_std = df_base_center['roc_auc'].std(ddof=1)
    
    enh_bal_mean = df_enh_seeds['balanced_accuracy'].mean()
    enh_bal_std = df_enh_seeds['balanced_accuracy'].std(ddof=1)
    base_bal_mean = df_base_center['balanced_accuracy'].mean()
    base_bal_std = df_base_center['balanced_accuracy'].std(ddof=1)
    
    paired_auc_diffs = df_enh_seeds['roc_auc'].values - df_base_center['roc_auc'].values
    paired_bal_diffs = df_enh_seeds['balanced_accuracy'].values - df_base_center['balanced_accuracy'].values
    
    combined_std_quad = np.sqrt(enh_auc_std**2 + base_auc_std**2)
    combined_std_sum = enh_auc_std + base_auc_std
    mean_auc_diff = enh_auc_mean - base_auc_mean
    
    # -------------------------------------------------------------
    # External Transfer on 13 Independent Records for Winning Setup
    # -------------------------------------------------------------
    print(f"\n--- Evaluating External Transfer on 13 Independent TPEHGT Records ---")
    scaler_final = StandardScaler()
    X_train_final = scaler_final.fit_transform(best_X)
    X_test_final = scaler_final.transform(df_tpehgt_burst[best_cand_cols].values)
    y_test_final = df_tpehgt_burst['label'].values
    
    if best_model_type == 'logistic_regression':
        final_model = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
        final_model.fit(X_train_final, y)
        probs_trans = final_model.predict_proba(X_test_final)[:, 1]
    else:
        lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
        svm = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', probability=True, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)
        cal_lr = CalibratedClassifierCV(estimator=lr, method='sigmoid', cv=3).fit(X_train_final, y)
        cal_svm = CalibratedClassifierCV(estimator=svm, method='sigmoid', cv=3).fit(X_train_final, y)
        cal_rf = CalibratedClassifierCV(estimator=rf, method='sigmoid', cv=3).fit(X_train_final, y)
        probs_trans = (cal_lr.predict_proba(X_test_final)[:, 1] + 
                       cal_svm.predict_proba(X_test_final)[:, 1] + 
                       cal_rf.predict_proba(X_test_final)[:, 1]) / 3.0
                       
    preds_trans = (probs_trans >= 0.50).astype(int)
    
    n_tot = len(y_test_final)
    n_p = int(np.sum(y_test_final))
    n_n = n_tot - n_p
    tp = int(np.sum((y_test_final == 1) & (preds_trans == 1)))
    tn = int(np.sum((y_test_final == 0) & (preds_trans == 0)))
    fp = int(np.sum((y_test_final == 0) & (preds_trans == 1)))
    fn = int(np.sum((y_test_final == 1) & (preds_trans == 0)))
    
    trans_acc = (tp + tn) / n_tot
    trans_bal = balanced_accuracy_score(y_test_final, preds_trans)
    trans_sens = tp / n_p if n_p > 0 else 0.0
    trans_spec = tn / n_n if n_n > 0 else 0.0
    trans_auc = roc_auc_score(y_test_final, probs_trans)
    
    w_acc_lo, w_acc_hi = wilson_score_interval(tp + tn, n_tot)
    w_sens_lo, w_sens_hi = wilson_score_interval(tp, n_p)
    w_spec_lo, w_spec_hi = wilson_score_interval(tn, n_n)
    
    transfer_summary = {
        'Configuration': best_cand_name,
        'Accuracy': f"{trans_acc:.3f} [{w_acc_lo:.3f}-{w_acc_hi:.3f}]*",
        'Balanced_Acc': f"{trans_bal:.3f} [Unreliable small-sample N=13]*",
        'Sensitivity': f"{trans_sens:.3f} [{w_sens_lo:.3f}-{w_sens_hi:.3f}]*",
        'Specificity': f"{trans_spec:.3f} [{w_spec_lo:.3f}-{w_spec_hi:.3f}]*",
        'ROC_AUC': f"{trans_auc:.3f}*",
        'True_Preterms': n_p,
        'Detected_Preterms': tp,
        'True_Terms': n_n,
        'Detected_Terms': tn
    }
    pd.DataFrame([transfer_summary]).to_csv(os.path.join(reports_dir, 'enhancements_transfer_tpehgt13.csv'), index=False)
    
    print("\n" + "="*70)
    print("CONSOLIDATED RESULTS SUMMARY")
    print("="*70)
    print(df_comp[['Step', 'Model', 'Feature_Count', 'Balanced_Acc', 'Sensitivity', 'Specificity', 'ROC_AUC']].to_string(index=False))
    print("-" * 70)
    print(f"Winning Enhancement across Steps 1-4: {best_cand_name}")
    print(f"10-Seed Variance vs Validated Baseline (Center 60s):")
    print(f"  Baseline ROC-AUC:    {base_auc_mean:.3f} +/- {base_auc_std:.3f}")
    print(f"  Enhancement ROC-AUC: {enh_auc_mean:.3f} +/- {enh_auc_std:.3f}")
    print(f"  Mean AUC Delta:      {mean_auc_diff:+.3f}")
    print(f"  Combined Std (Sum):  {combined_std_sum:.3f}")
    print(f"  Combined Std (Quad): {combined_std_quad:.3f}")
    print(f"  Paired Delta Mean:   {paired_auc_diffs.mean():+.3f} +/- {paired_auc_diffs.std(ddof=1):.3f}")
    print(f"  Paired Wins:         {sum(paired_auc_diffs > 0)} / 10 seeds")
    print(f"  Exceeds Noise Band (Sum)?  {mean_auc_diff > combined_std_sum}")
    print(f"  Exceeds Noise Band (Quad)? {mean_auc_diff > combined_std_quad}")
    print("-" * 70)
    print(f"External Transfer on N=13 Independent Records:")
    print(f"  Sensitivity: {trans_sens:.3f} [{w_sens_lo:.3f}-{w_sens_hi:.3f}]* (TP: {tp}/{n_p})")
    print(f"  Specificity: {trans_spec:.3f} [{w_spec_lo:.3f}-{w_spec_hi:.3f}]* (TN: {tn}/{n_n})")
    print(f"  Balanced Acc: {trans_bal:.3f}")
    print(f"  ROC-AUC:      {trans_auc:.3f}")
    print("="*70)


if __name__ == '__main__':
    run_all_enhancements()
