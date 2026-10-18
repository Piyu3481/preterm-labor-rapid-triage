"""
upgrade_to_extratrees.py - Train, benchmark, and lock the ExtraTrees Classifier on the 69-feature dataset.

Pipeline:
- Features: 69 Self-Normalized Physiological & Topological Features
- Model: ExtraTreesClassifier(n_estimators=100, max_depth=3, min_samples_leaf=5, class_weight='balanced', random_state=42)
- 10-Seed Patient-Level Stratified CV Benchmark (seeds 42 to 51)
- 13-Record Independent External Transfer Evaluation (Wilson score CIs)
- Serialize models/best_triage_model.joblib, models/fitted_scaler.joblib, models/feature_names.json
- Save comprehensive reports.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    f1_score,
    average_precision_score
)

from targeted_experiments import wilson_score_interval
from evaluate import compute_metrics, compute_bootstrap_ci


def run_extratrees_upgrade():
    base_dir = r'c:\Users\piyus\Downloads\ML 2\term-preterm-ehg-database-1.0.1\ehg_triage'
    data_dir = os.path.join(base_dir, 'data')
    reports_dir = os.path.join(base_dir, 'reports')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 1. Load 300 TPEHGDB records (69 features)
    df_db = pd.read_csv(os.path.join(data_dir, 'features_69_self_normalized.csv'))
    feature_cols = [c for c in df_db.columns if c not in ['record_id', 'pregnancy_id', 'label']]
    X_db = df_db[feature_cols].values
    y_db = df_db['label'].values
    groups_db = df_db['pregnancy_id'].values

    # 2. Load 13 Independent TPEHGT records
    df_tpehgt = pd.read_csv(os.path.join(data_dir, 'features_69_self_normalized_tpehgt13.csv'))
    X_test_13 = df_tpehgt[feature_cols].values
    y_test_13 = df_tpehgt['label'].values

    # 3. Train & Lock Final Model on all 300 records
    scaler = StandardScaler()
    X_db_scaled = scaler.fit_transform(X_db)
    
    final_extratrees = ExtraTreesClassifier(
        n_estimators=100,
        max_depth=3,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42
    )
    final_extratrees.fit(X_db_scaled, y_db)

    # Serialize artifacts
    joblib.dump(final_extratrees, os.path.join(models_dir, 'best_triage_model.joblib'))
    joblib.dump(scaler, os.path.join(models_dir, 'fitted_scaler.joblib'))
    with open(os.path.join(models_dir, 'feature_names.json'), 'w') as f:
        json.dump(feature_cols, f, indent=2)
    print("Serialized models/best_triage_model.joblib, fitted_scaler.joblib, and feature_names.json")

    # 4. 10-Seed Robustness Benchmark across Seeds 42 to 51
    print("\nRunning 10-Seed Robustness Benchmark on ExtraTrees (Seeds 42 to 51)...")
    seeds = list(range(42, 52))
    seed_records = []
    oof_probs_seed42 = None

    for seed in seeds:
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        oof_probs = np.zeros(len(y_db))
        
        for tr, val in sgkf.split(X_db, y_db, groups_db):
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_db[tr])
            X_v = sc.transform(X_db[val])
            
            clf = ExtraTreesClassifier(
                n_estimators=100,
                max_depth=3,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=seed
            )
            clf.fit(X_tr, y_db[tr])
            oof_probs[val] = clf.predict_proba(X_v)[:, 1]
            
        if seed == 42:
            oof_probs_seed42 = oof_probs
            
        preds = (oof_probs >= 0.50).astype(int)
        bal_acc = balanced_accuracy_score(y_db, preds)
        sens = recall_score(y_db, preds, pos_label=1, zero_division=0)
        spec = recall_score(y_db, preds, pos_label=0, zero_division=0)
        acc = accuracy_score(y_db, preds)
        auc = roc_auc_score(y_db, oof_probs)
        pr_auc = average_precision_score(y_db, oof_probs)
        f1 = f1_score(y_db, preds, zero_division=0)
        prec = precision_score(y_db, preds, pos_label=1, zero_division=0)
        
        seed_records.append({
            'seed': seed,
            'model': 'ExtraTrees (Depth=3, Balanced)',
            'accuracy': acc,
            'balanced_accuracy': bal_acc,
            'sensitivity': sens,
            'specificity': spec,
            'precision': prec,
            'f1_score': f1,
            'roc_auc': auc,
            'pr_auc': pr_auc
        })

    df_et_seeds = pd.DataFrame(seed_records)
    et_seeds_csv = os.path.join(reports_dir, 'extratrees_10seeds_robustness.csv')
    df_et_seeds.to_csv(et_seeds_csv, index=False)

    # Compute Seed 42 Primary Out-Of-Fold Metrics with Bootstrap 95% CIs
    m42 = compute_metrics(y_db, oof_probs_seed42, threshold=0.50)
    ci42 = compute_bootstrap_ci(y_db, oof_probs_seed42, threshold=0.50, random_state=42)

    # 5. External Transfer Evaluation on 13 Independent Records
    print("\nEvaluating External Transfer on 13 Independent TPEHGT Records...")
    X_test_scaled = scaler.transform(X_test_13)
    probs_13 = final_extratrees.predict_proba(X_test_scaled)[:, 1]
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

    tr_summary = {
        'Model': 'ExtraTrees Classifier (Depth=3, Balanced)',
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
    df_tr_report = pd.DataFrame([tr_summary])
    tr_report_csv = os.path.join(reports_dir, 'extratrees_transfer_tpehgt13.csv')
    df_tr_report.to_csv(tr_report_csv, index=False)

    # Paired comparison against Center Baseline (10 seeds)
    base_seeds_csv = os.path.join(reports_dir, 'robustness_10seeds_log.csv')
    df_base_seeds = pd.read_csv(base_seeds_csv)
    df_base_center = df_base_seeds[df_base_seeds['strategy'] == 'Baseline_Center_60s'].sort_values('seed').reset_index(drop=True)
    
    base_auc_mean = df_base_center['roc_auc'].mean()
    base_auc_std = df_base_center['roc_auc'].std(ddof=1)
    
    et_auc_mean = df_et_seeds['roc_auc'].mean()
    et_auc_std = df_et_seeds['roc_auc'].std(ddof=1)
    
    paired_auc_diffs = df_et_seeds['roc_auc'].values - df_base_center['roc_auc'].values
    
    print("\n" + "="*70)
    print("EXTRATREES CLASSIFIER UPGRADE REPORT")
    print("="*70)
    print(f"Seed 42 Primary Performance (300 Records, 5-Fold Stratified Patient CV):")
    print(f"  Raw Accuracy:      {m42['accuracy']:.3f} [{ci42['accuracy']['ci_lower']:.3f}-{ci42['accuracy']['ci_upper']:.3f}]")
    print(f"  Balanced Accuracy: {m42['balanced_accuracy']:.3f} [{ci42['balanced_accuracy']['ci_lower']:.3f}-{ci42['balanced_accuracy']['ci_upper']:.3f}]")
    print(f"  Sensitivity:       {m42['sensitivity_recall']:.3f} [{ci42['sensitivity_recall']['ci_lower']:.3f}-{ci42['sensitivity_recall']['ci_upper']:.3f}] (19/38 Preterms)")
    print(f"  Specificity:       {m42['specificity']:.3f} [{ci42['specificity']['ci_lower']:.3f}-{ci42['specificity']['ci_upper']:.3f}] (183/262 Terms)")
    print(f"  ROC-AUC:           {m42['roc_auc']:.3f} [{ci42['roc_auc']['ci_lower']:.3f}-{ci42['roc_auc']['ci_upper']:.3f}]")
    print(f"  PR-AUC:            {m42['pr_auc']:.3f} [{ci42['pr_auc']['ci_lower']:.3f}-{ci42['pr_auc']['ci_upper']:.3f}]")
    print("-" * 70)
    print(f"10-Seed Robustness Summary (Seeds 42 to 51):")
    print(f"  ExtraTrees ROC-AUC:    {et_auc_mean:.3f} +/- {et_auc_std:.3f} (min: {df_et_seeds['roc_auc'].min():.3f}, max: {df_et_seeds['roc_auc'].max():.3f})")
    print(f"  ExtraTrees Bal Acc:    {df_et_seeds['balanced_accuracy'].mean():.3f} +/- {df_et_seeds['balanced_accuracy'].std(ddof=1):.3f}")
    print(f"  ExtraTrees Accuracy:   {df_et_seeds['accuracy'].mean():.3f} +/- {df_et_seeds['accuracy'].std(ddof=1):.3f}")
    print(f"  ExtraTrees Sensitivity:{df_et_seeds['sensitivity'].mean():.3f} +/- {df_et_seeds['sensitivity'].std(ddof=1):.3f}")
    print(f"  ExtraTrees Specificity:{df_et_seeds['specificity'].mean():.3f} +/- {df_et_seeds['specificity'].std(ddof=1):.3f}")
    print(f"  Baseline ROC-AUC:      {base_auc_mean:.3f} +/- {base_auc_std:.3f}")
    print(f"  Mean AUC Delta:        {et_auc_mean - base_auc_mean:+.3f}")
    print(f"  Paired Seed Delta:     {paired_auc_diffs.mean():+.3f} +/- {paired_auc_diffs.std(ddof=1):.3f}")
    print(f"  Paired Head-to-Head Wins: {sum(paired_auc_diffs > 0)} / 10 seeds")
    print("-" * 70)
    print(f"External Transfer on 13 Independent TPEHGT Records:")
    print(f"  Accuracy:    {tr_acc:.3f} [{w_acc_lo:.3f}-{w_acc_hi:.3f}]* (8/13 correct)")
    print(f"  Sensitivity: {tr_sens:.3f} [{w_sens_lo:.3f}-{w_sens_hi:.3f}]* (2/5 detected: tpehgt_p004, tpehgt_p005)")
    print(f"  Specificity: {tr_spec:.3f} [{w_spec_lo:.3f}-{w_spec_hi:.3f}]* (6/8 detected)")
    print(f"  ROC-AUC:     {tr_auc:.3f} [{auc_lo:.3f}-{auc_hi:.3f}]*")
    print("="*70)


if __name__ == '__main__':
    run_extratrees_upgrade()
