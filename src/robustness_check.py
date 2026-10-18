"""
robustness_check.py - 10-Seed Multi-Run Variance & Robustness Benchmark

Evaluates:
1. Baseline: Original Unmodified Logistic Regression (Center 60s, 57 features, C=1.0)
2. Max-Energy: Original Unmodified Logistic Regression (Max-Energy 60s, 57 features, C=1.0)

Across 10 fixed random seeds: 42, 43, 44, 45, 46, 47, 48, 49, 50, 51.
For each seed:
- 5-Fold StratifiedGroupKFold on patient/pregnancy_id
- StandardScaler fit strictly inside training fold
- LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=seed)
- Collect OOF predictions and compute Balanced Accuracy, Sensitivity, Specificity, and ROC-AUC.
"""

import os
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


def evaluate_cohort_across_seeds(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seeds: list[int],
    cohort_name: str
) -> pd.DataFrame:
    records = []

    for seed in seeds:
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        oof_probs = np.zeros(len(y))
        
        for train_idx, val_idx in sgkf.split(X, y, groups):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            clf = LogisticRegression(
                C=1.0,
                class_weight='balanced',
                max_iter=1000,
                random_state=seed
            )
            clf.fit(X_train_scaled, y_train)
            oof_probs[val_idx] = clf.predict_proba(X_val_scaled)[:, 1]
            
        preds = (oof_probs >= 0.5).astype(int)
        
        bal_acc = balanced_accuracy_score(y, preds)
        sens = recall_score(y, preds, pos_label=1, zero_division=0)
        spec = recall_score(y, preds, pos_label=0, zero_division=0)
        acc = accuracy_score(y, preds)
        prec = precision_score(y, preds, pos_label=1, zero_division=0)
        f1 = f1_score(y, preds, pos_label=1, zero_division=0)
        roc_auc = roc_auc_score(y, oof_probs)
        pr_auc = average_precision_score(y, oof_probs)
        
        records.append({
            'strategy': cohort_name,
            'seed': seed,
            'balanced_accuracy': bal_acc,
            'sensitivity': sens,
            'specificity': spec,
            'roc_auc': roc_auc,
            'accuracy': acc,
            'precision': prec,
            'f1_score': f1,
            'pr_auc': pr_auc
        })
        
    return pd.DataFrame(records)


def run_robustness_benchmark():
    base_dir = r'c:\Users\piyus\Downloads\ML 2\term-preterm-ehg-database-1.0.1\ehg_triage'
    center_csv = os.path.join(base_dir, 'data', 'features_60s_center.csv')
    max_eng_csv = os.path.join(base_dir, 'data', 'features_60s_max_energy.csv')
    reports_dir = os.path.join(base_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    # 1. Load Center 60s
    df_center = pd.read_csv(center_csv)
    df_center = df_center[df_center['source'] == 'TPEHGDB'].reset_index(drop=True)
    f_center = [c for c in df_center.columns if c not in ['record_id', 'source', 'pregnancy_id', 'label', 'start_time_sec']]
    X_center = df_center[f_center].values
    y_center = df_center['label'].values
    groups_center = df_center['pregnancy_id'].values

    # 2. Load Max-Energy 60s
    df_max_eng = pd.read_csv(max_eng_csv)
    f_max_eng = [c for c in df_max_eng.columns if c not in ['record_id', 'pregnancy_id', 'label', 'start_time_sec']]
    X_max_eng = df_max_eng[f_max_eng].values
    y_max_eng = df_max_eng['label'].values
    groups_max_eng = df_max_eng['pregnancy_id'].values
    
    seeds = list(range(42, 52))  # 42 to 51 (10 seeds)
    
    print(f"Evaluating Baseline (Center 60s) across {len(seeds)} seeds: {seeds}")
    df_base_seeds = evaluate_cohort_across_seeds(X_center, y_center, groups_center, seeds, 'Baseline_Center_60s')
    
    print(f"Evaluating Max-Energy across {len(seeds)} seeds: {seeds}")
    df_max_seeds = evaluate_cohort_across_seeds(X_max_eng, y_max_eng, groups_max_eng, seeds, 'Max_Energy_60s')
    
    # Combine per-seed logs
    df_all_seeds = pd.concat([df_base_seeds, df_max_seeds], ignore_index=True)
    seed_log_csv = os.path.join(reports_dir, 'robustness_10seeds_log.csv')
    df_all_seeds.to_csv(seed_log_csv, index=False)
    print(f"Saved seed-by-seed log to: {seed_log_csv}")
    
    # Compute summary statistics
    metrics_to_summarize = ['balanced_accuracy', 'sensitivity', 'specificity', 'roc_auc', 'accuracy', 'pr_auc']
    
    summary_rows = []
    for name, df_sub in [('Baseline (Center 60s)', df_base_seeds), ('Max-Energy (Peak 60s)', df_max_seeds)]:
        row = {'Strategy': name}
        for m in metrics_to_summarize:
            mean_val = df_sub[m].mean()
            std_val = df_sub[m].std(ddof=1)
            row[f'{m}_mean'] = mean_val
            row[f'{m}_std'] = std_val
            row[f'{m}_display'] = f"{mean_val:.3f} +/- {std_val:.3f}"
        summary_rows.append(row)
        
    df_summary = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(reports_dir, 'robustness_10seeds_summary.csv')
    df_summary.to_csv(summary_csv, index=False)
    print(f"Saved summary to: {summary_csv}")
    
    # Statistical Noise Band Analysis
    base_auc_mean = df_base_seeds['roc_auc'].mean()
    base_auc_std = df_base_seeds['roc_auc'].std(ddof=1)
    
    max_auc_mean = df_max_seeds['roc_auc'].mean()
    max_auc_std = df_max_seeds['roc_auc'].std(ddof=1)
    
    auc_diff_mean = max_auc_mean - base_auc_mean
    combined_std_sum = base_auc_std + max_auc_std
    combined_std_quadrature = np.sqrt(base_auc_std**2 + max_auc_std**2)
    
    # Paired per-seed difference
    paired_diffs = df_max_seeds['roc_auc'].values - df_base_seeds['roc_auc'].values
    paired_diff_mean = paired_diffs.mean()
    paired_diff_std = paired_diffs.std(ddof=1)
    
    print("\n" + "="*70)
    print("10-SEED ROBUSTNESS & VARIANCE ANALYSIS SUMMARY")
    print("="*70)
    print(f"Baseline (Center 60s) Across 10 Seeds:")
    print(f"  Balanced Acc: {df_base_seeds['balanced_accuracy'].mean():.3f} +/- {df_base_seeds['balanced_accuracy'].std(ddof=1):.3f} (min: {df_base_seeds['balanced_accuracy'].min():.3f}, max: {df_base_seeds['balanced_accuracy'].max():.3f})")
    print(f"  Sensitivity:  {df_base_seeds['sensitivity'].mean():.3f} +/- {df_base_seeds['sensitivity'].std(ddof=1):.3f} (min: {df_base_seeds['sensitivity'].min():.3f}, max: {df_base_seeds['sensitivity'].max():.3f})")
    print(f"  ROC-AUC:      {base_auc_mean:.3f} +/- {base_auc_std:.3f} (min: {df_base_seeds['roc_auc'].min():.3f}, max: {df_base_seeds['roc_auc'].max():.3f})")
    print("-" * 70)
    print(f"Max-Energy (Peak 60s) Across 10 Seeds:")
    print(f"  Balanced Acc: {df_max_seeds['balanced_accuracy'].mean():.3f} +/- {df_max_seeds['balanced_accuracy'].std(ddof=1):.3f} (min: {df_max_seeds['balanced_accuracy'].min():.3f}, max: {df_max_seeds['balanced_accuracy'].max():.3f})")
    print(f"  Sensitivity:  {df_max_seeds['sensitivity'].mean():.3f} +/- {df_max_seeds['sensitivity'].std(ddof=1):.3f} (min: {df_max_seeds['sensitivity'].min():.3f}, max: {df_max_seeds['sensitivity'].max():.3f})")
    print(f"  ROC-AUC:      {max_auc_mean:.3f} +/- {max_auc_std:.3f} (min: {df_max_seeds['roc_auc'].min():.3f}, max: {df_max_seeds['roc_auc'].max():.3f})")
    print("-" * 70)
    print(f"ROC-AUC Comparison:")
    print(f"  Mean AUC Improvement:                  {auc_diff_mean:+.3f}")
    print(f"  Baseline Std (sigma_base):             {base_auc_std:.3f}")
    print(f"  Max-Energy Std (sigma_max):            {max_auc_std:.3f}")
    print(f"  Sum of Std Devs (sigma_base + sigma_max): {combined_std_sum:.3f}")
    print(f"  Quadrature Combined Std (sqrt(s1^2 + s2^2)): {combined_std_quadrature:.3f}")
    print(f"  Paired Seed Delta (Mean +/- Std):      {paired_diff_mean:+.3f} +/- {paired_diff_std:.3f}")
    print(f"  Paired Wins for Max-Energy:            {sum(paired_diffs > 0)} / 10 seeds")
    print(f"  Does Mean AUC Improvement exceed Combined Std (Sum)? {auc_diff_mean > combined_std_sum}")
    print(f"  Does Mean AUC Improvement exceed Combined Std (Quadrature)? {auc_diff_mean > combined_std_quadrature}")
    print("="*70)

    # Save detailed analysis
    analysis_dict = {
        'baseline_auc_mean': base_auc_mean,
        'baseline_auc_std': base_auc_std,
        'max_energy_auc_mean': max_auc_mean,
        'max_energy_auc_std': max_auc_std,
        'auc_improvement': auc_diff_mean,
        'combined_std_sum': combined_std_sum,
        'combined_std_quadrature': combined_std_quadrature,
        'exceeds_combined_std_sum': bool(auc_diff_mean > combined_std_sum),
        'exceeds_combined_std_quadrature': bool(auc_diff_mean > combined_std_quadrature),
        'paired_diff_mean': paired_diff_mean,
        'paired_diff_std': paired_diff_std,
        'paired_wins': int(sum(paired_diffs > 0))
    }
    with open(os.path.join(reports_dir, 'robustness_analysis.json'), 'w') as f:
        import json
        json.dump(analysis_dict, f, indent=2)


if __name__ == '__main__':
    run_robustness_benchmark()
