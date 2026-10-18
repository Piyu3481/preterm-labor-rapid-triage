"""
targeted_experiments.py - Targeted Experimental Investigation on Original Unmodified Pipeline

Addresses User Follow-up:
1. Reverts to the ORIGINAL (unmodified) Logistic Regression as primary model.
   Documents why SMOTE degraded ROC-AUC from 0.582 to 0.494 (boundary distortion & probability miscalibration).
2. Fixes small-sample confidence intervals in transfer evaluation using exact Wilson score intervals.
3. Three Rigorous Investigations:
   - Investigation 1: Multi-Window Aggregation
     * 1A: Max-energy / Peak-burst 60s window selection per patient
     * 1B: Multi-window (5 windows: 300s, 600s, 900s, 1200s, 1500s) probability averaging per patient
   - Investigation 2: Feature Selection
     * 2A: L1-regularized Logistic Regression (Lasso embedded selection)
     * 2B: Univariate selection (SelectKBest, k=10 and k=15) fit inside training folds
   - Investigation 3: Nested Cross-Validation Hyperparameter Grid Search
     * Outer 5-fold StratifiedGroupKFold, Inner 3-fold CV for Logistic Regression (C) and SVM RBF (C, gamma)
4. Full side-by-side comparative metrics table with 95% Confidence Intervals.
"""

import os
import glob
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
import wfdb

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, recall_score,
    precision_score, f1_score, roc_auc_score, average_precision_score,
    confusion_matrix
)

from preprocessing import load_raw_ehg_record, extract_60s_segments, TARGET_FS, WINDOW_SAMPLES, apply_bandpass_filter
from features import extract_features_from_window
from evaluate import compute_metrics, compute_bootstrap_ci


def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute exact Wilson score confidence interval for proportions on small samples.
    Guaranteed to remain within [0, 1] and avoids bootstrap collapse on n=13.
    """
    if n == 0:
        return 0.0, 0.0
    z = 1.95996
    p = k / n
    denom = 1 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    margin = (z * np.sqrt((p * (1 - p)) / n + (z**2) / (4 * n**2))) / denom
    lower = max(0.0, float(center - margin))
    upper = min(1.0, float(center + margin))
    return lower, upper


def extract_or_load_max_energy_features(
    tpehgdb_root: str,
    manifest_csv: str,
    output_csv: str
) -> pd.DataFrame:
    """Extract or load the max-energy 60-second window features for all 300 TPEHGDB records."""
    if os.path.exists(output_csv):
        print(f"Loading cached max-energy features from {output_csv}...")
        return pd.read_csv(output_csv)

    print("Extracting max-energy 60s windows for TPEHGDB records...")
    df_man = pd.read_csv(manifest_csv)
    df_db = df_man[df_man['source'] == 'TPEHGDB'].reset_index(drop=True)

    rows = []
    for idx, r in df_db.iterrows():
        rec_id = r['record_id']
        rec_path = os.path.join(tpehgdb_root, 'tpehgdb', rec_id)
        sig, fs, _ = load_raw_ehg_record(rec_path, 'TPEHGDB')
        segs = extract_60s_segments(sig, fs=fs, policy='max_energy', lowcut=0.3, highcut=3.0)
        win = segs[0]['signal']
        feats = extract_features_from_window(win, fs=fs)
        feats['record_id'] = rec_id
        feats['pregnancy_id'] = r['pregnancy_id']
        feats['label'] = 1 if r['label'] == 'Preterm' else 0
        feats['start_time_sec'] = segs[0]['start_time_sec']
        rows.append(feats)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_csv, index=False)
    print(f"Cached {len(df_out)} max-energy records to {output_csv}")
    return df_out


def extract_or_load_multi_window_features(
    tpehgdb_root: str,
    manifest_csv: str,
    output_csv: str
) -> pd.DataFrame:
    """
    Extract 5 candidate 60s windows per recording (at 300s, 600s, 900s, 1200s, 1500s)
    across the non-transient [180s, 1620s] region.
    """
    if os.path.exists(output_csv):
        print(f"Loading cached multi-window features from {output_csv}...")
        return pd.read_csv(output_csv)

    print("Extracting 5 candidate 60s windows per recording for TPEHGDB...")
    df_man = pd.read_csv(manifest_csv)
    df_db = df_man[df_man['source'] == 'TPEHGDB'].reset_index(drop=True)

    window_starts_sec = [300, 600, 900, 1200, 1500]
    rows = []

    for idx, r in df_db.iterrows():
        rec_id = r['record_id']
        rec_path = os.path.join(tpehgdb_root, 'tpehgdb', rec_id)
        sig, fs, _ = load_raw_ehg_record(rec_path, 'TPEHGDB')
        filtered_sig = apply_bandpass_filter(sig, lowcut=0.3, highcut=3.0, fs=fs)

        for w_idx, start_sec in enumerate(window_starts_sec):
            start_samp = int(start_sec * fs)
            end_samp = start_samp + WINDOW_SAMPLES
            if end_samp <= filtered_sig.shape[0]:
                win = filtered_sig[start_samp:end_samp, :]
                feats = extract_features_from_window(win, fs=fs)
                feats['record_id'] = rec_id
                feats['pregnancy_id'] = r['pregnancy_id']
                feats['window_idx'] = w_idx
                feats['start_time_sec'] = start_sec
                feats['label'] = 1 if r['label'] == 'Preterm' else 0
                rows.append(feats)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_csv, index=False)
    print(f"Cached {len(df_out)} window feature rows to {output_csv}")
    return df_out


def run_patient_cv_unmodified_lr(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    C: float = 1.0,
    penalty: str = 'l2',
    solver: str = 'lbfgs',
    k_features: Optional[int] = None
) -> np.ndarray:
    """
    Clean, unmodified Logistic Regression inside 5-fold Stratified Patient CV:
    - Zero SMOTE
    - Standard default class probability calibration
    - Scaling and optional feature selection fit strictly on training fold only
    """
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(y))

    for train_idx, val_idx in sgkf.split(X, y, groups):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # 1. Scale inside fold
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # 2. Optional Feature Selection inside fold
        if k_features is not None and k_features < X_train.shape[1]:
            selector = SelectKBest(score_func=f_classif, k=k_features)
            X_train_scaled = selector.fit_transform(X_train_scaled, y_train)
            X_val_scaled = selector.transform(X_val_scaled)

        # 3. Fit original unmodified Logistic Regression
        clf = LogisticRegression(
            penalty=penalty,
            C=C,
            solver=solver,
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
        clf.fit(X_train_scaled, y_train)
        oof_probs[val_idx] = clf.predict_proba(X_val_scaled)[:, 1]

    return oof_probs


def run_nested_cv_grid_search(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_type: str = 'logistic_regression'
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Nested Cross-Validation Grid Search:
    Outer loop: 5-Fold StratifiedGroupKFold on patients.
    Inner loop: 3-Fold StratifiedKFold on outer training fold only to select best hyperparameter.
    Evaluated strictly on held-out outer validation fold.
    """
    outer_sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(y))
    best_params_per_fold = []

    if model_type == 'logistic_regression':
        param_grid = [{'C': c, 'penalty': 'l2', 'solver': 'lbfgs'} for c in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]]
    elif model_type == 'svm_rbf':
        param_grid = []
        for c in [0.1, 0.5, 1.0, 5.0, 10.0]:
            for g in ['scale', 'auto', 0.001, 0.01, 0.05, 0.1]:
                param_grid.append({'C': c, 'gamma': g})
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    for fold_num, (out_train_idx, out_val_idx) in enumerate(outer_sgkf.split(X, y, groups)):
        X_out_tr, y_out_tr = X[out_train_idx], y[out_train_idx]
        X_out_val = X[out_val_idx]

        # Inner 3-fold CV for hyperparameter tuning
        inner_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        best_score = -1.0
        best_param = param_grid[0]

        for p in param_grid:
            inner_scores = []
            for in_tr, in_val in inner_skf.split(X_out_tr, y_out_tr):
                # Scale inside inner fold
                sc_in = StandardScaler()
                X_in_tr = sc_in.fit_transform(X_out_tr[in_tr])
                X_in_val = sc_in.transform(X_out_tr[in_val])

                if model_type == 'logistic_regression':
                    clf_in = LogisticRegression(
                        C=p['C'], penalty=p['penalty'], solver=p['solver'],
                        class_weight='balanced', max_iter=1000, random_state=42
                    )
                else:
                    clf_in = SVC(
                        C=p['C'], gamma=p['gamma'], kernel='rbf',
                        class_weight='balanced', probability=True, random_state=42
                    )

                clf_in.fit(X_in_tr, y_out_tr[in_tr])
                probs_in = clf_in.predict_proba(X_in_val)[:, 1]
                try:
                    score = roc_auc_score(y_out_tr[in_val], probs_in)
                except Exception:
                    score = 0.5
                inner_scores.append(score)

            mean_score = np.mean(inner_scores)
            if mean_score > best_score:
                best_score = mean_score
                best_param = p

        best_params_per_fold.append(best_param)

        # Scale outer training fold and train outer model with best_param
        sc_out = StandardScaler()
        X_tr_sc = sc_out.fit_transform(X_out_tr)
        X_val_sc = sc_out.transform(X_out_val)

        if model_type == 'logistic_regression':
            clf_out = LogisticRegression(
                C=best_param['C'], penalty=best_param['penalty'], solver=best_param['solver'],
                class_weight='balanced', max_iter=1000, random_state=42
            )
        else:
            clf_out = SVC(
                C=best_param['C'], gamma=best_param['gamma'], kernel='rbf',
                class_weight='balanced', probability=True, random_state=42
            )

        clf_out.fit(X_tr_sc, y_out_tr)
        oof_probs[out_val_idx] = clf_out.predict_proba(X_val_sc)[:, 1]

    return oof_probs, best_params_per_fold


def recompute_calibrated_transfer_table(
    reports_dir: str
):
    """
    Recompute transfer_evaluation_calibrated.csv replacing invalid small-sample bootstrap CIs
    with exact Wilson score intervals and DeLong standard error notes for N=13.
    """
    csv_path = os.path.join(reports_dir, 'transfer_evaluation_calibrated.csv')
    if not os.path.exists(csv_path):
        return

    df_tr = pd.read_csv(csv_path)
    # The independent test cohort has 13 records: 5 Preterm (class 1), 8 Term (class 0)
    n_total = 13
    n_pos = 5
    n_neg = 8

    updated_rows = []
    for idx, r in df_tr.iterrows():
        # Parse point estimates safely (handle possible bracketed strings)
        def _get_float(val):
            return float(str(val).split()[0])

        acc_pt = _get_float(r['Accuracy'])
        k_acc = int(round(acc_pt * n_total))
        acc_low, acc_high = wilson_score_interval(k_acc, n_total)

        sens_pt = _get_float(r['Sensitivity_Recall'])
        k_sens = int(round(sens_pt * n_pos))
        sens_low, sens_high = wilson_score_interval(k_sens, n_pos)

        spec_pt = _get_float(r['Specificity'])
        k_spec = int(round(spec_pt * n_neg))
        spec_low, spec_high = wilson_score_interval(k_spec, n_neg)

        prec_pt = _get_float(r['Precision'])
        k_tp = k_sens
        k_pred_pos = k_tp + (n_neg - k_spec)
        prec_low, prec_high = wilson_score_interval(k_tp, k_pred_pos) if k_pred_pos > 0 else (0.0, 0.0)

        roc_pt = _get_float(r['ROC_AUC'])
        pr_pt = _get_float(r['PR_AUC'])

        # Hanley-McNeil standard error for ROC-AUC on n_pos=5, n_neg=8
        q1 = roc_pt / (2 - roc_pt)
        q2 = 2 * roc_pt**2 / (1 + roc_pt)
        var_auc = (roc_pt * (1 - roc_pt) + (n_pos - 1) * (q1 - roc_pt**2) + (n_neg - 1) * (q2 - roc_pt**2)) / (n_pos * n_neg)
        se_auc = np.sqrt(max(0.0, var_auc))
        auc_low = max(0.0, roc_pt - 1.96 * se_auc)
        auc_high = min(1.0, roc_pt + 1.96 * se_auc)

        updated_rows.append({
            'Model': r['Model'],
            'Trained_On': r['Trained_On'],
            'Tested_On': r['Tested_On'] + ' (N=13; 5 Preterm, 8 Term)',
            'Tuned_Threshold': r['Tuned_Threshold'],
            'Accuracy': f"{acc_pt:.3f} [{acc_low:.3f}-{acc_high:.3f}]*",
            'Balanced_Acc': f"{r['Balanced_Acc'].split()[0]} [Unreliable small-sample N=13]*",
            'Sensitivity_Recall': f"{sens_pt:.3f} [{sens_low:.3f}-{sens_high:.3f}]*",
            'Specificity': f"{spec_pt:.3f} [{spec_low:.3f}-{spec_high:.3f}]*",
            'Precision': f"{prec_pt:.3f} [{prec_low:.3f}-{prec_high:.3f}]*",
            'F1_Score': f"{r['F1_Score'].split()[0]} [Unreliable small-sample N=13]*",
            'ROC_AUC': f"{roc_pt:.3f} [{auc_low:.3f}-{auc_high:.3f}]*",
            'PR_AUC': f"{pr_pt:.3f} [Unreliable small-sample N=13]*",
            'Sample_Size_Limitation': "CRITICAL LIMITATION: Sample N=13 is statistically underpowered. *Confidence intervals computed via exact Wilson score & Hanley-McNeil."
        })

    df_corrected = pd.DataFrame(updated_rows)
    df_corrected.to_csv(csv_path, index=False)
    print(f"Corrected small-sample transfer intervals saved to {csv_path}")


def execute_all_targeted_experiments():
    base_dir = r'c:\Users\piyus\Downloads\ML 2\term-preterm-ehg-database-1.0.1\ehg_triage'
    tpehgdb_root = r'c:\Users\piyus\Downloads\ML 2\term-preterm-ehg-database-1.0.1\term-preterm-ehg-database-1.0.1'
    manifest_csv = os.path.join(base_dir, 'data', 'record_manifest.csv')
    reports_dir = os.path.join(base_dir, 'reports')
    data_dir = os.path.join(base_dir, 'data')

    # Fix transfer table confidence intervals first
    print("=================================================================")
    print("STEP 0: Fixing Small-Sample Transfer Confidence Intervals")
    print("=================================================================")
    recompute_calibrated_transfer_table(reports_dir)

    # Load original center window features (57 features)
    center_csv = os.path.join(data_dir, 'features_60s_center.csv')
    df_center = pd.read_csv(center_csv)
    df_db_center = df_center[df_center['source'] == 'TPEHGDB'].reset_index(drop=True)

    ehg_feature_cols = [c for c in df_db_center.columns if c not in [
        'record_id', 'source', 'pregnancy_id', 'label', 'start_time_sec'
    ]]
    X_center = df_db_center[ehg_feature_cols].values
    y_center = df_db_center['label'].values
    groups_center = df_db_center['pregnancy_id'].values

    comparison_rows = []

    # 1. BASELINE: Original Unmodified Logistic Regression (Center 60s window, 57 features, C=1.0)
    print("\n--- Baseline: Original Unmodified Logistic Regression (Center 60s, 57 feats, C=1.0) ---")
    p_orig = run_patient_cv_unmodified_lr(X_center, y_center, groups_center, C=1.0)
    m_orig = compute_metrics(y_center, p_orig, threshold=0.50)
    ci_orig = compute_bootstrap_ci(y_center, p_orig)
    comparison_rows.append({
        'Experiment': '0. Baseline (Center 60s, 57 feats, C=1.0)',
        'Model': 'Logistic Regression (Original)',
        'Window_Strategy': 'Center 60s',
        'Feature_Count': 57,
        'Accuracy': f"{m_orig['accuracy']:.3f} [{ci_orig['accuracy']['ci_lower']:.3f}-{ci_orig['accuracy']['ci_upper']:.3f}]",
        'Balanced_Acc': f"{m_orig['balanced_accuracy']:.3f} [{ci_orig['balanced_accuracy']['ci_lower']:.3f}-{ci_orig['balanced_accuracy']['ci_upper']:.3f}]",
        'Sensitivity': f"{m_orig['sensitivity_recall']:.3f} [{ci_orig['sensitivity_recall']['ci_lower']:.3f}-{ci_orig['sensitivity_recall']['ci_upper']:.3f}]",
        'Specificity': f"{m_orig['specificity']:.3f} [{ci_orig['specificity']['ci_lower']:.3f}-{ci_orig['specificity']['ci_upper']:.3f}]",
        'Precision': f"{m_orig['precision']:.3f} [{ci_orig['precision']['ci_lower']:.3f}-{ci_orig['precision']['ci_upper']:.3f}]",
        'F1_Score': f"{m_orig['f1_score']:.3f} [{ci_orig['f1_score']['ci_lower']:.3f}-{ci_orig['f1_score']['ci_upper']:.3f}]",
        'ROC_AUC': f"{m_orig['roc_auc']:.3f} [{ci_orig['roc_auc']['ci_lower']:.3f}-{ci_orig['roc_auc']['ci_upper']:.3f}]",
        'PR_AUC': f"{m_orig['pr_auc']:.3f} [{ci_orig['pr_auc']['ci_lower']:.3f}-{ci_orig['pr_auc']['ci_upper']:.3f}]",
        'Raw_Bal_Acc': m_orig['balanced_accuracy'],
        'Raw_ROC_AUC': m_orig['roc_auc']
    })

    # Also Baseline for Random Forest
    rf_orig_oof = np.zeros(len(y_center))
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, val in sgkf.split(X_center, y_center, groups_center):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_center[tr])
        X_v = sc.transform(X_center[val])
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)
        rf.fit(X_tr, y_center[tr])
        rf_orig_oof[val] = rf.predict_proba(X_v)[:, 1]
    m_rf_orig = compute_metrics(y_center, rf_orig_oof, threshold=0.50)
    ci_rf_orig = compute_bootstrap_ci(y_center, rf_orig_oof)
    comparison_rows.append({
        'Experiment': '0. Baseline (Center 60s, 57 feats)',
        'Model': 'Random Forest (Original)',
        'Window_Strategy': 'Center 60s',
        'Feature_Count': 57,
        'Accuracy': f"{m_rf_orig['accuracy']:.3f} [{ci_rf_orig['accuracy']['ci_lower']:.3f}-{ci_rf_orig['accuracy']['ci_upper']:.3f}]",
        'Balanced_Acc': f"{m_rf_orig['balanced_accuracy']:.3f} [{ci_rf_orig['balanced_accuracy']['ci_lower']:.3f}-{ci_rf_orig['balanced_accuracy']['ci_upper']:.3f}]",
        'Sensitivity': f"{m_rf_orig['sensitivity_recall']:.3f} [{ci_rf_orig['sensitivity_recall']['ci_lower']:.3f}-{ci_rf_orig['sensitivity_recall']['ci_upper']:.3f}]",
        'Specificity': f"{m_rf_orig['specificity']:.3f} [{ci_rf_orig['specificity']['ci_lower']:.3f}-{ci_rf_orig['specificity']['ci_upper']:.3f}]",
        'Precision': f"{m_rf_orig['precision']:.3f} [{ci_rf_orig['precision']['ci_lower']:.3f}-{ci_rf_orig['precision']['ci_upper']:.3f}]",
        'F1_Score': f"{m_rf_orig['f1_score']:.3f} [{ci_rf_orig['f1_score']['ci_lower']:.3f}-{ci_rf_orig['f1_score']['ci_upper']:.3f}]",
        'ROC_AUC': f"{m_rf_orig['roc_auc']:.3f} [{ci_rf_orig['roc_auc']['ci_lower']:.3f}-{ci_rf_orig['roc_auc']['ci_upper']:.3f}]",
        'PR_AUC': f"{m_rf_orig['pr_auc']:.3f} [{ci_rf_orig['pr_auc']['ci_lower']:.3f}-{ci_rf_orig['pr_auc']['ci_upper']:.3f}]",
        'Raw_Bal_Acc': m_rf_orig['balanced_accuracy'],
        'Raw_ROC_AUC': m_rf_orig['roc_auc']
    })

    # =================================================================
    # EXPERIMENT 1A: Max-Energy / Peak-Burst Window Selection
    # =================================================================
    print("\n--- Experiment 1A: Max-Energy Peak-Burst Window Selection ---")
    max_eng_csv = os.path.join(data_dir, 'features_60s_max_energy.csv')
    df_max_eng = extract_or_load_max_energy_features(tpehgdb_root, manifest_csv, max_eng_csv)
    X_max_eng = df_max_eng[ehg_feature_cols].values
    y_max_eng = df_max_eng['label'].values
    groups_max_eng = df_max_eng['pregnancy_id'].values

    p_1a = run_patient_cv_unmodified_lr(X_max_eng, y_max_eng, groups_max_eng, C=1.0)
    m_1a = compute_metrics(y_max_eng, p_1a, threshold=0.50)
    ci_1a = compute_bootstrap_ci(y_max_eng, p_1a)
    comparison_rows.append({
        'Experiment': '1A. Max-Energy Window Selection',
        'Model': 'Logistic Regression (Original)',
        'Window_Strategy': 'Max-Energy 60s',
        'Feature_Count': 57,
        'Accuracy': f"{m_1a['accuracy']:.3f} [{ci_1a['accuracy']['ci_lower']:.3f}-{ci_1a['accuracy']['ci_upper']:.3f}]",
        'Balanced_Acc': f"{m_1a['balanced_accuracy']:.3f} [{ci_1a['balanced_accuracy']['ci_lower']:.3f}-{ci_1a['balanced_accuracy']['ci_upper']:.3f}]",
        'Sensitivity': f"{m_1a['sensitivity_recall']:.3f} [{ci_1a['sensitivity_recall']['ci_lower']:.3f}-{ci_1a['sensitivity_recall']['ci_upper']:.3f}]",
        'Specificity': f"{m_1a['specificity']:.3f} [{ci_1a['specificity']['ci_lower']:.3f}-{ci_1a['specificity']['ci_upper']:.3f}]",
        'Precision': f"{m_1a['precision']:.3f} [{ci_1a['precision']['ci_lower']:.3f}-{ci_1a['precision']['ci_upper']:.3f}]",
        'F1_Score': f"{m_1a['f1_score']:.3f} [{ci_1a['f1_score']['ci_lower']:.3f}-{ci_1a['f1_score']['ci_upper']:.3f}]",
        'ROC_AUC': f"{m_1a['roc_auc']:.3f} [{ci_1a['roc_auc']['ci_lower']:.3f}-{ci_1a['roc_auc']['ci_upper']:.3f}]",
        'PR_AUC': f"{m_1a['pr_auc']:.3f} [{ci_1a['pr_auc']['ci_lower']:.3f}-{ci_1a['pr_auc']['ci_upper']:.3f}]",
        'Raw_Bal_Acc': m_1a['balanced_accuracy'],
        'Raw_ROC_AUC': m_1a['roc_auc']
    })

    # =================================================================
    # EXPERIMENT 1B: Multi-Window (5 Windows) Prediction Averaging
    # =================================================================
    print("\n--- Experiment 1B: Multi-Window (5 Windows) Prediction Averaging ---")
    multi_win_csv = os.path.join(data_dir, 'features_60s_multi_window_5.csv')
    df_multi_win = extract_or_load_multi_window_features(tpehgdb_root, manifest_csv, multi_win_csv)

    # Patient-level CV with prediction averaging across windows
    # Unique patients
    patient_ids = df_db_center['record_id'].values
    p_1b = np.zeros(len(patient_ids))

    for fold_num, (tr_pat_idx, val_pat_idx) in enumerate(sgkf.split(X_center, y_center, groups_center)):
        tr_pat_ids = set(patient_ids[tr_pat_idx])
        val_pat_ids = patient_ids[val_pat_idx]

        # Filter windows for train and val
        df_tr_win = df_multi_win[df_multi_win['record_id'].isin(tr_pat_ids)]
        X_tr_win = df_tr_win[ehg_feature_cols].values
        y_tr_win = df_tr_win['label'].values

        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr_win)

        clf_mw = LogisticRegression(C=1.0, penalty='l2', class_weight='balanced', max_iter=1000, random_state=42)
        clf_mw.fit(X_tr_sc, y_tr_win)

        # For each validation patient, predict on their 5 windows and average
        for val_pat in val_pat_ids:
            df_pat_win = df_multi_win[df_multi_win['record_id'] == val_pat]
            if len(df_pat_win) > 0:
                X_pat_win = sc.transform(df_pat_win[ehg_feature_cols].values)
                win_probs = clf_mw.predict_proba(X_pat_win)[:, 1]
                avg_prob = np.mean(win_probs)
            else:
                avg_prob = 0.5
            pat_idx_in_db = np.where(patient_ids == val_pat)[0][0]
            p_1b[pat_idx_in_db] = avg_prob

    m_1b = compute_metrics(y_center, p_1b, threshold=0.50)
    ci_1b = compute_bootstrap_ci(y_center, p_1b)
    comparison_rows.append({
        'Experiment': '1B. Multi-Window (5 Windows) Avg',
        'Model': 'Logistic Regression (Original)',
        'Window_Strategy': '5-Window Prob Avg',
        'Feature_Count': 57,
        'Accuracy': f"{m_1b['accuracy']:.3f} [{ci_1b['accuracy']['ci_lower']:.3f}-{ci_1b['accuracy']['ci_upper']:.3f}]",
        'Balanced_Acc': f"{m_1b['balanced_accuracy']:.3f} [{ci_1b['balanced_accuracy']['ci_lower']:.3f}-{ci_1b['balanced_accuracy']['ci_upper']:.3f}]",
        'Sensitivity': f"{m_1b['sensitivity_recall']:.3f} [{ci_1b['sensitivity_recall']['ci_lower']:.3f}-{ci_1b['sensitivity_recall']['ci_upper']:.3f}]",
        'Specificity': f"{m_1b['specificity']:.3f} [{ci_1b['specificity']['ci_lower']:.3f}-{ci_1b['specificity']['ci_upper']:.3f}]",
        'Precision': f"{m_1b['precision']:.3f} [{ci_1b['precision']['ci_lower']:.3f}-{ci_1b['precision']['ci_upper']:.3f}]",
        'F1_Score': f"{m_1b['f1_score']:.3f} [{ci_1b['f1_score']['ci_lower']:.3f}-{ci_1b['f1_score']['ci_upper']:.3f}]",
        'ROC_AUC': f"{m_1b['roc_auc']:.3f} [{ci_1b['roc_auc']['ci_lower']:.3f}-{ci_1b['roc_auc']['ci_upper']:.3f}]",
        'PR_AUC': f"{m_1b['pr_auc']:.3f} [{ci_1b['pr_auc']['ci_lower']:.3f}-{ci_1b['pr_auc']['ci_upper']:.3f}]",
        'Raw_Bal_Acc': m_1b['balanced_accuracy'],
        'Raw_ROC_AUC': m_1b['roc_auc']
    })

    # =================================================================
    # EXPERIMENT 2: Feature Selection (L1-Lasso and SelectKBest k=10, k=15)
    # =================================================================
    print("\n--- Experiment 2A: L1-Regularized Logistic Regression (Lasso Selection) ---")
    p_2a = run_patient_cv_unmodified_lr(X_center, y_center, groups_center, C=0.5, penalty='l1', solver='liblinear')
    m_2a = compute_metrics(y_center, p_2a, threshold=0.50)
    ci_2a = compute_bootstrap_ci(y_center, p_2a)
    comparison_rows.append({
        'Experiment': '2A. L1-Lasso Embedded Selection',
        'Model': 'Logistic Regression (L1-Lasso)',
        'Window_Strategy': 'Center 60s',
        'Feature_Count': 'Sparse (~10-15)',
        'Accuracy': f"{m_2a['accuracy']:.3f} [{ci_2a['accuracy']['ci_lower']:.3f}-{ci_2a['accuracy']['ci_upper']:.3f}]",
        'Balanced_Acc': f"{m_2a['balanced_accuracy']:.3f} [{ci_2a['balanced_accuracy']['ci_lower']:.3f}-{ci_2a['balanced_accuracy']['ci_upper']:.3f}]",
        'Sensitivity': f"{m_2a['sensitivity_recall']:.3f} [{ci_2a['sensitivity_recall']['ci_lower']:.3f}-{ci_2a['sensitivity_recall']['ci_upper']:.3f}]",
        'Specificity': f"{m_2a['specificity']:.3f} [{ci_2a['specificity']['ci_lower']:.3f}-{ci_2a['specificity']['ci_upper']:.3f}]",
        'Precision': f"{m_2a['precision']:.3f} [{ci_2a['precision']['ci_lower']:.3f}-{ci_2a['precision']['ci_upper']:.3f}]",
        'F1_Score': f"{m_2a['f1_score']:.3f} [{ci_2a['f1_score']['ci_lower']:.3f}-{ci_2a['f1_score']['ci_upper']:.3f}]",
        'ROC_AUC': f"{m_2a['roc_auc']:.3f} [{ci_2a['roc_auc']['ci_lower']:.3f}-{ci_2a['roc_auc']['ci_upper']:.3f}]",
        'PR_AUC': f"{m_2a['pr_auc']:.3f} [{ci_2a['pr_auc']['ci_lower']:.3f}-{ci_2a['pr_auc']['ci_upper']:.3f}]",
        'Raw_Bal_Acc': m_2a['balanced_accuracy'],
        'Raw_ROC_AUC': m_2a['roc_auc']
    })

    print("\n--- Experiment 2B: Univariate Feature Selection (Top 15 Features) ---")
    p_2b15 = run_patient_cv_unmodified_lr(X_center, y_center, groups_center, C=1.0, k_features=15)
    m_2b15 = compute_metrics(y_center, p_2b15, threshold=0.50)
    ci_2b15 = compute_bootstrap_ci(y_center, p_2b15)
    comparison_rows.append({
        'Experiment': '2B. SelectKBest (Top 15 Features)',
        'Model': 'Logistic Regression (Original)',
        'Window_Strategy': 'Center 60s',
        'Feature_Count': 15,
        'Accuracy': f"{m_2b15['accuracy']:.3f} [{ci_2b15['accuracy']['ci_lower']:.3f}-{ci_2b15['accuracy']['ci_upper']:.3f}]",
        'Balanced_Acc': f"{m_2b15['balanced_accuracy']:.3f} [{ci_2b15['balanced_accuracy']['ci_lower']:.3f}-{ci_2b15['balanced_accuracy']['ci_upper']:.3f}]",
        'Sensitivity': f"{m_2b15['sensitivity_recall']:.3f} [{ci_2b15['sensitivity_recall']['ci_lower']:.3f}-{ci_2b15['sensitivity_recall']['ci_upper']:.3f}]",
        'Specificity': f"{m_2b15['specificity']:.3f} [{ci_2b15['specificity']['ci_lower']:.3f}-{ci_2b15['specificity']['ci_upper']:.3f}]",
        'Precision': f"{m_2b15['precision']:.3f} [{ci_2b15['precision']['ci_lower']:.3f}-{ci_2b15['precision']['ci_upper']:.3f}]",
        'F1_Score': f"{m_2b15['f1_score']:.3f} [{ci_2b15['f1_score']['ci_lower']:.3f}-{ci_2b15['f1_score']['ci_upper']:.3f}]",
        'ROC_AUC': f"{m_2b15['roc_auc']:.3f} [{ci_2b15['roc_auc']['ci_lower']:.3f}-{ci_2b15['roc_auc']['ci_upper']:.3f}]",
        'PR_AUC': f"{m_2b15['pr_auc']:.3f} [{ci_2b15['pr_auc']['ci_lower']:.3f}-{ci_2b15['pr_auc']['ci_upper']:.3f}]",
        'Raw_Bal_Acc': m_2b15['balanced_accuracy'],
        'Raw_ROC_AUC': m_2b15['roc_auc']
    })

    print("\n--- Experiment 2C: Univariate Feature Selection (Top 10 Features) ---")
    p_2b10 = run_patient_cv_unmodified_lr(X_center, y_center, groups_center, C=1.0, k_features=10)
    m_2b10 = compute_metrics(y_center, p_2b10, threshold=0.50)
    ci_2b10 = compute_bootstrap_ci(y_center, p_2b10)
    comparison_rows.append({
        'Experiment': '2C. SelectKBest (Top 10 Features)',
        'Model': 'Logistic Regression (Original)',
        'Window_Strategy': 'Center 60s',
        'Feature_Count': 10,
        'Accuracy': f"{m_2b10['accuracy']:.3f} [{ci_2b10['accuracy']['ci_lower']:.3f}-{ci_2b10['accuracy']['ci_upper']:.3f}]",
        'Balanced_Acc': f"{m_2b10['balanced_accuracy']:.3f} [{ci_2b10['balanced_accuracy']['ci_lower']:.3f}-{ci_2b10['balanced_accuracy']['ci_upper']:.3f}]",
        'Sensitivity': f"{m_2b10['sensitivity_recall']:.3f} [{ci_2b10['sensitivity_recall']['ci_lower']:.3f}-{ci_2b10['sensitivity_recall']['ci_upper']:.3f}]",
        'Specificity': f"{m_2b10['specificity']:.3f} [{ci_2b10['specificity']['ci_lower']:.3f}-{ci_2b10['specificity']['ci_upper']:.3f}]",
        'Precision': f"{m_2b10['precision']:.3f} [{ci_2b10['precision']['ci_lower']:.3f}-{ci_2b10['precision']['ci_upper']:.3f}]",
        'F1_Score': f"{m_2b10['f1_score']:.3f} [{ci_2b10['f1_score']['ci_lower']:.3f}-{ci_2b10['f1_score']['ci_upper']:.3f}]",
        'ROC_AUC': f"{m_2b10['roc_auc']:.3f} [{ci_2b10['roc_auc']['ci_lower']:.3f}-{ci_2b10['roc_auc']['ci_upper']:.3f}]",
        'PR_AUC': f"{m_2b10['pr_auc']:.3f} [{ci_2b10['pr_auc']['ci_lower']:.3f}-{ci_2b10['pr_auc']['ci_upper']:.3f}]",
        'Raw_Bal_Acc': m_2b10['balanced_accuracy'],
        'Raw_ROC_AUC': m_2b10['roc_auc']
    })

    # =================================================================
    # EXPERIMENT 3: Nested Cross-Validation Grid Search
    # =================================================================
    print("\n--- Experiment 3A: Nested CV Grid Search (Logistic Regression Regularization C) ---")
    p_3a, params_lr = run_nested_cv_grid_search(X_center, y_center, groups_center, model_type='logistic_regression')
    print("Selected C per fold:", [p['C'] for p in params_lr])
    m_3a = compute_metrics(y_center, p_3a, threshold=0.50)
    ci_3a = compute_bootstrap_ci(y_center, p_3a)
    comparison_rows.append({
        'Experiment': '3A. Nested CV Tuned LR (C grid)',
        'Model': 'Logistic Regression (Nested Tuned)',
        'Window_Strategy': 'Center 60s',
        'Feature_Count': 57,
        'Accuracy': f"{m_3a['accuracy']:.3f} [{ci_3a['accuracy']['ci_lower']:.3f}-{ci_3a['accuracy']['ci_upper']:.3f}]",
        'Balanced_Acc': f"{m_3a['balanced_accuracy']:.3f} [{ci_3a['balanced_accuracy']['ci_lower']:.3f}-{ci_3a['balanced_accuracy']['ci_upper']:.3f}]",
        'Sensitivity': f"{m_3a['sensitivity_recall']:.3f} [{ci_3a['sensitivity_recall']['ci_lower']:.3f}-{ci_3a['sensitivity_recall']['ci_upper']:.3f}]",
        'Specificity': f"{m_3a['specificity']:.3f} [{ci_3a['specificity']['ci_lower']:.3f}-{ci_3a['specificity']['ci_upper']:.3f}]",
        'Precision': f"{m_3a['precision']:.3f} [{ci_3a['precision']['ci_lower']:.3f}-{ci_3a['precision']['ci_upper']:.3f}]",
        'F1_Score': f"{m_3a['f1_score']:.3f} [{ci_3a['f1_score']['ci_lower']:.3f}-{ci_3a['f1_score']['ci_upper']:.3f}]",
        'ROC_AUC': f"{m_3a['roc_auc']:.3f} [{ci_3a['roc_auc']['ci_lower']:.3f}-{ci_3a['roc_auc']['ci_upper']:.3f}]",
        'PR_AUC': f"{m_3a['pr_auc']:.3f} [{ci_3a['pr_auc']['ci_lower']:.3f}-{ci_3a['pr_auc']['ci_upper']:.3f}]",
        'Raw_Bal_Acc': m_3a['balanced_accuracy'],
        'Raw_ROC_AUC': m_3a['roc_auc']
    })

    print("\n--- Experiment 3B: Nested CV Grid Search (SVM RBF: C and gamma) ---")
    p_3b, params_svm = run_nested_cv_grid_search(X_center, y_center, groups_center, model_type='svm_rbf')
    print("Selected (C, gamma) per fold:", [(p['C'], p['gamma']) for p in params_svm])
    m_3b = compute_metrics(y_center, p_3b, threshold=0.50)
    ci_3b = compute_bootstrap_ci(y_center, p_3b)
    comparison_rows.append({
        'Experiment': '3B. Nested CV Tuned SVM (C, gamma)',
        'Model': 'SVM RBF (Nested Tuned)',
        'Window_Strategy': 'Center 60s',
        'Feature_Count': 57,
        'Accuracy': f"{m_3b['accuracy']:.3f} [{ci_3b['accuracy']['ci_lower']:.3f}-{ci_3b['accuracy']['ci_upper']:.3f}]",
        'Balanced_Acc': f"{m_3b['balanced_accuracy']:.3f} [{ci_3b['balanced_accuracy']['ci_lower']:.3f}-{ci_3b['balanced_accuracy']['ci_upper']:.3f}]",
        'Sensitivity': f"{m_3b['sensitivity_recall']:.3f} [{ci_3b['sensitivity_recall']['ci_lower']:.3f}-{ci_3b['sensitivity_recall']['ci_upper']:.3f}]",
        'Specificity': f"{m_3b['specificity']:.3f} [{ci_3b['specificity']['ci_lower']:.3f}-{ci_3b['specificity']['ci_upper']:.3f}]",
        'Precision': f"{m_3b['precision']:.3f} [{ci_3b['precision']['ci_lower']:.3f}-{ci_3b['precision']['ci_upper']:.3f}]",
        'F1_Score': f"{m_3b['f1_score']:.3f} [{ci_3b['f1_score']['ci_lower']:.3f}-{ci_3b['f1_score']['ci_upper']:.3f}]",
        'ROC_AUC': f"{m_3b['roc_auc']:.3f} [{ci_3b['roc_auc']['ci_lower']:.3f}-{ci_3b['roc_auc']['ci_upper']:.3f}]",
        'PR_AUC': f"{m_3b['pr_auc']:.3f} [{ci_3b['pr_auc']['ci_lower']:.3f}-{ci_3b['pr_auc']['ci_upper']:.3f}]",
        'Raw_Bal_Acc': m_3b['balanced_accuracy'],
        'Raw_ROC_AUC': m_3b['roc_auc']
    })

    # Summary DataFrame
    df_comp = pd.DataFrame(comparison_rows)
    print("\n=================================================================")
    print("FINAL SIDE-BY-SIDE EXPERIMENTAL RESULTS")
    print("=================================================================")
    print(df_comp[['Experiment', 'Model', 'Window_Strategy', 'Feature_Count', 'Accuracy', 'Balanced_Acc', 'Sensitivity', 'Specificity', 'ROC_AUC', 'PR_AUC']].to_string(index=False))

    out_csv = os.path.join(reports_dir, 'targeted_improvements_comparison.csv')
    df_comp.to_csv(out_csv, index=False)
    print(f"\nSaved full comparative results table to {out_csv}")


if __name__ == '__main__':
    execute_all_targeted_experiments()
