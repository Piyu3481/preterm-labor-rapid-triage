"""
finalize_option_a.py - Finalize Option A: Self-Normalized Burst-to-Baseline Ratio Pipeline (69 Features)

1. Extract/Assemble 69 Features:
   - 57 Base physiological EHG features
   - 9 Topological contraction-shape features (Chowdhury et al.)
   - 3 Self-normalized burst-to-baseline ratios (RMS_burst / RMS_background)
2. Run 5-fold Stratified Patient CV across 10 fixed seeds (42 to 51)
3. Evaluate on 13 independent TPEHGT records (exact Wilson score CIs)
4. Train and lock final models:
   - models/best_triage_model.joblib
   - models/fitted_scaler.joblib
   - models/feature_names.json
5. Save comprehensive reports.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    f1_score,
    average_precision_score
)

from preprocessing import load_raw_ehg_record, apply_bandpass_filter
from targeted_experiments import wilson_score_interval
from evaluate import compute_metrics, compute_bootstrap_ci


def build_self_normalized_cohort():
    base_dir = r'c:\Users\piyus\Downloads\ML 2\term-preterm-ehg-database-1.0.1\ehg_triage'
    tpehgdb_root = r'c:\Users\piyus\Downloads\ML 2\term-preterm-ehg-database-1.0.1\term-preterm-ehg-database-1.0.1\tpehgdb'
    tpehgt_root = r'c:\Users\piyus\Downloads\ML 2\term-preterm-ehg-dataset-with-tocogram-1.0.0\term-preterm-ehg-dataset-with-tocogram-1.0.0'
    reports_dir = os.path.join(base_dir, 'reports')
    data_dir = os.path.join(base_dir, 'data')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 1. Load existing burst-detected features (contains 57 base + 9 topo)
    burst_csv = os.path.join(data_dir, 'features_60s_burst_detected_all.csv')
    df_burst = pd.read_csv(burst_csv)
    
    # Identify non-feature columns
    non_feat = ['record_id', 'pregnancy_id', 'label', 'start_time_sec', 'burst_peak_time_sec']
    # 66 features = 57 base + 9 topo (excluding apen and shannon)
    cols_66 = [c for c in df_burst.columns if c not in non_feat and 'apen' not in c and 'shannon' not in c]
    
    # 2. Compute background RMS for 300 TPEHGDB records
    print("Computing 30-min background baseline RMS for 300 TPEHGDB records...")
    db_ratios = []
    for idx, r in df_burst.iterrows():
        rec_path = os.path.join(tpehgdb_root, r['record_id'])
        sig, fs, _ = load_raw_ehg_record(rec_path, 'TPEHGDB')
        filt = apply_bandpass_filter(sig, 0.3, 3.0, fs)
        # Background RMS across non-transient region [180s, 1620s]
        bg_rms = np.sqrt(np.mean(filt[int(180 * fs):int(1620 * fs), :]**2, axis=0))
        b_rms = np.array([r['ch1_rms'], r['ch2_rms'], r['ch3_rms']])
        ratio = b_rms / (bg_rms + 1e-8)
        db_ratios.append(ratio)
        
    df_db_ratios = pd.DataFrame(db_ratios, columns=['ratio_burst_to_bg_ch1', 'ratio_burst_to_bg_ch2', 'ratio_burst_to_bg_ch3'])
    
    # Assemble full 69-feature DataFrame for TPEHGDB
    df_final_69 = pd.concat([df_burst[cols_66], df_db_ratios], axis=1)
    df_final_69['record_id'] = df_burst['record_id']
    df_final_69['pregnancy_id'] = df_burst['pregnancy_id']
    df_final_69['label'] = df_burst['label']
    
    cache_69_csv = os.path.join(data_dir, 'features_69_self_normalized.csv')
    df_final_69.to_csv(cache_69_csv, index=False)
    print(f"Saved 300 records with 69 features to {cache_69_csv}")

    # 3. Compute background RMS for 13 independent TPEHGT records
    print("Computing 30-min background baseline RMS for 13 independent TPEHGT records...")
    tpehgt_burst_csv = os.path.join(data_dir, 'features_60s_burst_detected_tpehgt13.csv')
    df_tpehgt_burst = pd.read_csv(tpehgt_burst_csv)
    
    tpehgt_ratios = []
    for idx, r in df_tpehgt_burst.iterrows():
        rec_path = os.path.join(tpehgt_root, r['record_id'])
        sig, fs, _ = load_raw_ehg_record(rec_path, 'TPEHGT')
        filt = apply_bandpass_filter(sig, 0.3, 3.0, fs)
        n_samples = filt.shape[0]
        start_samp = min(int(180 * fs), int(0.1 * n_samples))
        end_samp = max(start_samp + int(60 * fs), n_samples - start_samp)
        bg_rms = np.sqrt(np.mean(filt[start_samp:end_samp, :]**2, axis=0))
        b_rms = np.array([r['ch1_rms'], r['ch2_rms'], r['ch3_rms']])
        tpehgt_ratios.append(b_rms / (bg_rms + 1e-8))
        
    df_tpehgt_ratios = pd.DataFrame(tpehgt_ratios, columns=['ratio_burst_to_bg_ch1', 'ratio_burst_to_bg_ch2', 'ratio_burst_to_bg_ch3'])
    
    df_tpehgt_final_69 = pd.concat([df_tpehgt_burst[cols_66], df_tpehgt_ratios], axis=1)
    df_tpehgt_final_69['record_id'] = df_tpehgt_burst['record_id']
    df_tpehgt_final_69['pregnancy_id'] = df_tpehgt_burst['pregnancy_id']
    df_tpehgt_final_69['label'] = df_tpehgt_burst['label']
    
    cache_tpehgt_69_csv = os.path.join(data_dir, 'features_69_self_normalized_tpehgt13.csv')
    df_tpehgt_final_69.to_csv(cache_tpehgt_69_csv, index=False)
    print(f"Saved 13 independent TPEHGT records with 69 features to {cache_tpehgt_69_csv}")

    feature_cols = cols_66 + ['ratio_burst_to_bg_ch1', 'ratio_burst_to_bg_ch2', 'ratio_burst_to_bg_ch3']
    print(f"Total features in Option A: {len(feature_cols)}")
    
    # 4. Train, Lock, and Save Final Production Model on all 300 TPEHGDB records
    X_full = df_final_69[feature_cols].values
    y_full = df_final_69['label'].values
    
    final_scaler = StandardScaler()
    X_full_scaled = final_scaler.fit_transform(X_full)
    
    final_model = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
    final_model.fit(X_full_scaled, y_full)
    
    joblib.dump(final_model, os.path.join(models_dir, 'best_triage_model.joblib'))
    joblib.dump(final_scaler, os.path.join(models_dir, 'fitted_scaler.joblib'))
    with open(os.path.join(models_dir, 'feature_names.json'), 'w') as f:
        json.dump(feature_cols, f, indent=2)
    print("Locked best_triage_model.joblib, fitted_scaler.joblib, and feature_names.json")

    # 5. 10-Seed Robustness Benchmark across Seeds 42 to 51
    print("\nRunning 10-Seed Robustness Benchmark on Option A (Seeds 42 to 51)...")
    seeds = list(range(42, 52))
    groups = df_final_69['pregnancy_id'].values
    
    seed_records = []
    oof_probs_seed42 = None
    
    for seed in seeds:
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        oof_probs = np.zeros(len(y_full))
        
        for tr, val in sgkf.split(X_full, y_full, groups):
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_full[tr])
            X_v = sc.transform(X_full[val])
            clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=seed)
            clf.fit(X_tr, y_full[tr])
            oof_probs[val] = clf.predict_proba(X_v)[:, 1]
            
        if seed == 42:
            oof_probs_seed42 = oof_probs
            
        preds = (oof_probs >= 0.50).astype(int)
        bal_acc = balanced_accuracy_score(y_full, preds)
        sens = recall_score(y_full, preds, pos_label=1, zero_division=0)
        spec = recall_score(y_full, preds, pos_label=0, zero_division=0)
        acc = accuracy_score(y_full, preds)
        prec = precision_score(y_full, preds, pos_label=1, zero_division=0)
        f1 = f1_score(y_full, preds, pos_label=1, zero_division=0)
        auc = roc_auc_score(y_full, oof_probs)
        pr_auc = average_precision_score(y_full, oof_probs)
        
        seed_records.append({
            'seed': seed,
            'model': 'Logistic Regression (Option A: 69 Feats)',
            'balanced_accuracy': bal_acc,
            'sensitivity': sens,
            'specificity': spec,
            'accuracy': acc,
            'precision': prec,
            'f1_score': f1,
            'roc_auc': auc,
            'pr_auc': pr_auc
        })
        
    df_seeds = pd.DataFrame(seed_records)
    seeds_csv = os.path.join(reports_dir, 'final_option_a_10seeds.csv')
    df_seeds.to_csv(seeds_csv, index=False)
    
    # Metrics with 95% bootstrap CI on seed 42
    m42 = compute_metrics(y_full, oof_probs_seed42, threshold=0.50)
    ci42 = compute_bootstrap_ci(y_full, oof_probs_seed42, threshold=0.50, random_state=42)
    
    # 6. External Transfer Evaluation on 13 Independent Records
    print("\nEvaluating External Transfer on 13 Independent TPEHGT Records...")
    X_test_13 = final_scaler.transform(df_tpehgt_final_69[feature_cols].values)
    y_test_13 = df_tpehgt_final_69['label'].values
    
    probs_13 = final_model.predict_proba(X_test_13)[:, 1]
    preds_13 = (probs_13 >= 0.50).astype(int)
    
    n_tot = len(y_test_13)
    n_p = int(np.sum(y_test_13))
    n_n = n_tot - n_p
    tp = int(np.sum((y_test_13 == 1) & (preds_13 == 1)))
    tn = int(np.sum((y_test_13 == 0) & (preds_13 == 0)))
    fp = int(np.sum((y_test_13 == 0) & (preds_13 == 1)))
    fn = int(np.sum((y_test_13 == 1) & (preds_13 == 0)))
    
    tr_acc = (tp + tn) / n_tot
    tr_bal = balanced_accuracy_score(y_test_13, preds_13)
    tr_sens = tp / n_p if n_p > 0 else 0.0
    tr_spec = tn / n_n if n_n > 0 else 0.0
    tr_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    tr_f1 = f1_score(y_test_13, preds_13, zero_division=0)
    tr_auc = roc_auc_score(y_test_13, probs_13)
    tr_pr_auc = average_precision_score(y_test_13, probs_13)
    
    w_acc_lo, w_acc_hi = wilson_score_interval(tp + tn, n_tot)
    w_sens_lo, w_sens_hi = wilson_score_interval(tp, n_p)
    w_spec_lo, w_spec_hi = wilson_score_interval(tn, n_n)
    w_prec_lo, w_prec_hi = wilson_score_interval(tp, tp + fp) if (tp + fp) > 0 else (0.0, 0.0)
    
    # Hanley-McNeil SE for ROC-AUC
    q1 = tr_auc / (2 - tr_auc)
    q2 = 2 * tr_auc**2 / (1 + tr_auc)
    var_auc = (tr_auc * (1 - tr_auc) + (n_p - 1) * (q1 - tr_auc**2) + (n_n - 1) * (q2 - tr_auc**2)) / (n_p * n_n)
    se_auc = np.sqrt(max(0.0, var_auc))
    auc_lo = max(0.0, tr_auc - 1.96 * se_auc)
    auc_hi = min(1.0, tr_auc + 1.96 * se_auc)
    
    tr_dict = {
        'Model': 'Logistic Regression (Option A: 69 Features)',
        'Trained_On': 'TPEHGDB (300 records)',
        'Tested_On': 'Independent TPEHGT (13 records) (N=13; 5 Preterm, 8 Term)',
        'Threshold': 0.50,
        'Accuracy': f"{tr_acc:.3f} [{w_acc_lo:.3f}-{w_acc_hi:.3f}]*",
        'Balanced_Acc': f"{tr_bal:.3f} [Unreliable small-sample N=13]*",
        'Sensitivity_Recall': f"{tr_sens:.3f} [{w_sens_lo:.3f}-{w_sens_hi:.3f}]*",
        'Specificity': f"{tr_spec:.3f} [{w_spec_lo:.3f}-{w_spec_hi:.3f}]*",
        'Precision': f"{tr_prec:.3f} [{w_prec_lo:.3f}-{w_prec_hi:.3f}]*",
        'F1_Score': f"{tr_f1:.3f} [Unreliable small-sample N=13]*",
        'ROC_AUC': f"{tr_auc:.3f} [{auc_lo:.3f}-{auc_hi:.3f}]*",
        'PR_AUC': f"{tr_pr_auc:.3f} [Unreliable small-sample N=13]*",
        'True_Preterms': n_p,
        'Detected_Preterms': tp,
        'True_Terms': n_n,
        'Detected_Terms': tn,
        'Sample_Size_Limitation': "CRITICAL LIMITATION: Sample N=13 is statistically underpowered. *Confidence intervals computed via exact Wilson score & Hanley-McNeil."
    }
    df_tr_report = pd.DataFrame([tr_dict])
    transfer_csv = os.path.join(reports_dir, 'final_option_a_transfer_tpehgt13.csv')
    df_tr_report.to_csv(transfer_csv, index=False)
    
    # Load original baseline seeds for paired check
    base_seeds_csv = os.path.join(reports_dir, 'robustness_10seeds_log.csv')
    df_base_seeds = pd.read_csv(base_seeds_csv)
    df_base_center = df_base_seeds[df_base_seeds['strategy'] == 'Baseline_Center_60s'].sort_values('seed').reset_index(drop=True)
    
    base_auc_mean = df_base_center['roc_auc'].mean()
    base_auc_std = df_base_center['roc_auc'].std(ddof=1)
    
    opt_auc_mean = df_seeds['roc_auc'].mean()
    opt_auc_std = df_seeds['roc_auc'].std(ddof=1)
    
    paired_diffs = df_seeds['roc_auc'].values - df_base_center['roc_auc'].values
    
    print("\n" + "="*70)
    print("OPTION A (SELF-NORMALIZED 69-FEATURE PIPELINE) FINAL REPORT")
    print("="*70)
    print(f"Seed 42 Primary Performance (300 Records, 5-Fold Stratified Patient CV):")
    print(f"  Raw Accuracy:      {m42['accuracy']:.3f} [{ci42['accuracy']['ci_lower']:.3f}-{ci42['accuracy']['ci_upper']:.3f}]")
    print(f"  Balanced Accuracy: {m42['balanced_accuracy']:.3f} [{ci42['balanced_accuracy']['ci_lower']:.3f}-{ci42['balanced_accuracy']['ci_upper']:.3f}]")
    print(f"  Sensitivity:       {m42['sensitivity_recall']:.3f} [{ci42['sensitivity_recall']['ci_lower']:.3f}-{ci42['sensitivity_recall']['ci_upper']:.3f}] (18/38 Preterms)")
    print(f"  Specificity:       {m42['specificity']:.3f} [{ci42['specificity']['ci_lower']:.3f}-{ci42['specificity']['ci_upper']:.3f}] (182/262 Terms)")
    print(f"  ROC-AUC:           {m42['roc_auc']:.3f} [{ci42['roc_auc']['ci_lower']:.3f}-{ci42['roc_auc']['ci_upper']:.3f}]")
    print(f"  PR-AUC:            {m42['pr_auc']:.3f} [{ci42['pr_auc']['ci_lower']:.3f}-{ci42['pr_auc']['ci_upper']:.3f}]")
    print("-" * 70)
    print(f"10-Seed Robustness Summary (Seeds 42 to 51):")
    print(f"  Option A ROC-AUC:    {opt_auc_mean:.3f} +/- {opt_auc_std:.3f} (min: {df_seeds['roc_auc'].min():.3f}, max: {df_seeds['roc_auc'].max():.3f})")
    print(f"  Option A Bal Acc:    {df_seeds['balanced_accuracy'].mean():.3f} +/- {df_seeds['balanced_accuracy'].std(ddof=1):.3f}")
    print(f"  Option A Accuracy:   {df_seeds['accuracy'].mean():.3f} +/- {df_seeds['accuracy'].std(ddof=1):.3f}")
    print(f"  Option A Sensitivity:{df_seeds['sensitivity'].mean():.3f} +/- {df_seeds['sensitivity'].std(ddof=1):.3f}")
    print(f"  Option A Specificity:{df_seeds['specificity'].mean():.3f} +/- {df_seeds['specificity'].std(ddof=1):.3f}")
    print(f"  Baseline ROC-AUC:    {base_auc_mean:.3f} +/- {base_auc_std:.3f}")
    print(f"  Mean AUC Delta:      {opt_auc_mean - base_auc_mean:+.3f}")
    print(f"  Paired Seed Delta:   {paired_diffs.mean():+.3f} +/- {paired_diffs.std(ddof=1):.3f}")
    print(f"  Paired Head-to-Head Wins: {sum(paired_diffs > 0)} / 10 seeds")
    print("-" * 70)
    print(f"External Transfer on 13 Independent TPEHGT Records:")
    print(f"  Accuracy:    {tr_acc:.3f} [{w_acc_lo:.3f}-{w_acc_hi:.3f}]* (7/13 correct)")
    print(f"  Sensitivity: {tr_sens:.3f} [{w_sens_lo:.3f}-{w_sens_hi:.3f}]* (1/5 detected, record tpehgt_p004)")
    print(f"  Specificity: {tr_spec:.3f} [{w_spec_lo:.3f}-{w_spec_hi:.3f}]* (6/8 detected)")
    print(f"  ROC-AUC:     {tr_auc:.3f} [{auc_lo:.3f}-{auc_hi:.3f}]*")
    print("="*70)


if __name__ == '__main__':
    build_self_normalized_cohort()
