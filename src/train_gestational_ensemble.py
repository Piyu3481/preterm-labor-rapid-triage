"""
train_gestational_ensemble.py - Train and Validate Gestational-Conditioned Soft Ensembles.

Implements:
1. Extraction/matching of recording gestational age (rectime) from TPEHGDB headers.
2. Training of Tier 1: Specialized Late-Gestation Ensemble (rectime >= 26 weeks, N=138).
3. Training of Tier 2: General Screening Ensemble (all records, N=300).
4. Rigorous 5-fold Stratified Cross-Validation on both tiers.
5. External transfer validation on 13 independent TPEHGT records.
6. Model locking and artifact serialization in models/ directory.
7. Logging full benchmark results to reports/gestational_ensemble_evaluation.csv.
"""

import os
import sys
import glob
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    roc_auc_score,
    precision_score,
    f1_score,
    confusion_matrix
)

sys.path.append(os.path.dirname(__file__))
from targeted_experiments import wilson_score_interval


def parse_rectime_metadata(tpehgdb_root: str) -> pd.DataFrame:
    """Parse gestational age at recording time (rectime) from .hea files."""
    records = []
    for hea_path in glob.glob(os.path.join(tpehgdb_root, '*.hea')):
        rec_id = os.path.basename(hea_path).replace('.hea', '')
        rectime = 30.0  # default median fallback
        with open(hea_path, 'r') as f:
            for line in f:
                if 'Rectime' in line:
                    try:
                        rectime = float(line.strip('# \r\n').split()[1])
                    except Exception:
                        pass
        records.append({'record_id': rec_id, 'rectime': rectime})
    return pd.DataFrame(records)


def parse_tpehgt_rectime(tpehgt_root: str) -> pd.DataFrame:
    """Parse gestational age at recording time for TPEHGT records."""
    records = []
    for hea_path in glob.glob(os.path.join(tpehgt_root, '*.hea')):
        rec_id = os.path.basename(hea_path).replace('.hea', '')
        rectime = 30.0
        with open(hea_path, 'r') as f:
            for line in f:
                if 'Rectime' in line or 'Gestation at recording' in line:
                    try:
                        rectime = float(line.strip('# \r\n').split()[-1])
                    except Exception:
                        pass
        records.append({'record_id': rec_id, 'rectime': rectime})
    return pd.DataFrame(records)


def build_soft_ensemble(random_state: int = 42, weights: list = [2, 1]) -> VotingClassifier:
    """Construct calibrated Soft Voting Ensemble combining ExtraTrees and L2 Logistic Regression."""
    clf_et = ExtraTreesClassifier(
        n_estimators=100,
        max_depth=3,
        min_samples_leaf=3,
        class_weight='balanced',
        random_state=random_state
    )
    clf_lr = LogisticRegression(
        C=0.5,
        class_weight='balanced',
        max_iter=1000,
        random_state=random_state
    )
    ensemble = VotingClassifier(
        estimators=[('et', clf_et), ('lr', clf_lr)],
        voting='soft',
        weights=weights
    )
    return ensemble


def evaluate_cohort_cv(X: np.ndarray, y: np.ndarray, cv, random_state: int = 42):
    """Run 5-fold stratified CV for the soft ensemble and return metrics and predictions."""
    preds = np.zeros(len(y))
    probs = np.zeros(len(y))
    
    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_va_sc = scaler.transform(X_va)
        
        model = build_soft_ensemble(random_state=random_state)
        model.fit(X_tr_sc, y_tr)
        
        preds[val_idx] = model.predict(X_va_sc)
        probs[val_idx] = model.predict_proba(X_va_sc)[:, 1]
        
    acc = accuracy_score(y, preds)
    bal_acc = balanced_accuracy_score(y, preds)
    sens = recall_score(y, preds, pos_label=1)
    spec = recall_score(y, preds, pos_label=0)
    prec = precision_score(y, preds, pos_label=1, zero_division=0)
    f1 = f1_score(y, preds, pos_label=1, zero_division=0)
    auc = roc_auc_score(y, probs)
    cm = confusion_matrix(y, preds)
    
    return {
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'sensitivity': sens,
        'specificity': spec,
        'precision': prec,
        'f1': f1,
        'roc_auc': auc,
        'confusion_matrix': cm,
        'probs': probs,
        'preds': preds
    }


def main():
    base_dir = r'c:\Users\piyus\Downloads\ML 2\term-preterm-ehg-database-1.0.1\ehg_triage'
    tpehgdb_root = r'c:\Users\piyus\Downloads\ML 2\term-preterm-ehg-database-1.0.1\term-preterm-ehg-database-1.0.1\tpehgdb'
    tpehgt_root = r'c:\Users\piyus\Downloads\ML 2\term-preterm-ehg-dataset-with-tocogram-1.0.0\term-preterm-ehg-dataset-with-tocogram-1.0.0'
    data_dir = os.path.join(base_dir, 'data')
    models_dir = os.path.join(base_dir, 'models')
    reports_dir = os.path.join(base_dir, 'reports')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    # 1. Load 69 Self-Normalized Features
    df_feat = pd.read_csv(os.path.join(data_dir, 'features_69_self_normalized.csv'))
    df_rectime = parse_rectime_metadata(tpehgdb_root)
    df_merged = pd.merge(df_feat, df_rectime, on='record_id')
    
    meta_cols = ['record_id', 'pregnancy_id', 'label', 'rectime']
    feature_names = [c for c in df_feat.columns if c not in meta_cols]
    
    print(f"Total Cohort: {len(df_merged)} records with {len(feature_names)} self-normalized features.")
    
    # 2. Split into Sub-Cohorts
    # Tier 1: Late Gestation (>= 26.0 weeks, 3rd Trimester Labor Triage)
    df_late = df_merged[df_merged['rectime'] >= 26.0].reset_index(drop=True)
    # Tier 2: General Cohort (All records, Universal Screening)
    df_general = df_merged.copy()
    
    print(f"Tier 1 (Late Cohort >= 26 wks): N={len(df_late)} (Preterm: {sum(df_late.label==1)}, Term: {sum(df_late.label==0)})")
    print(f"Tier 2 (General Cohort All):    N={len(df_general)} (Preterm: {sum(df_general.label==1)}, Term: {sum(df_general.label==0)})")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Evaluate Tier 1: Late Gestation Soft Ensemble
    X_late = df_late[feature_names].values
    y_late = df_late['label'].values
    res_late = evaluate_cohort_cv(X_late, y_late, cv, random_state=42)
    
    # Evaluate Tier 2: General Screening Soft Ensemble
    X_gen = df_general[feature_names].values
    y_gen = df_general['label'].values
    res_gen = evaluate_cohort_cv(X_gen, y_gen, cv, random_state=42)
    
    print("\n=================== 5-FOLD STRATIFIED CV RESULTS ===================")
    print(f"Tier 1 (Late-Gestation Labor Triage, >= 26 wks, N={len(df_late)}):")
    print(f"  Accuracy:          {res_late['accuracy']*100:.2f}%")
    print(f"  Balanced Accuracy: {res_late['balanced_accuracy']*100:.2f}%")
    print(f"  Preterm Recall:    {res_late['sensitivity']*100:.2f}% ({int(res_late['sensitivity']*sum(y_late==1))}/{sum(y_late==1)})")
    print(f"  Term Specificity:  {res_late['specificity']*100:.2f}% ({int(res_late['specificity']*sum(y_late==0))}/{sum(y_late==0)})")
    print(f"  ROC-AUC:           {res_late['roc_auc']:.4f}")
    
    print(f"\nTier 2 (General Universal Screening, All Patients, N={len(df_general)}):")
    print(f"  Accuracy:          {res_gen['accuracy']*100:.2f}%")
    print(f"  Balanced Accuracy: {res_gen['balanced_accuracy']*100:.2f}%")
    print(f"  Preterm Recall:    {res_gen['sensitivity']*100:.2f}% ({int(res_gen['sensitivity']*sum(y_gen==1))}/{sum(y_gen==1)})")
    print(f"  Term Specificity:  {res_gen['specificity']*100:.2f}% ({int(res_gen['specificity']*sum(y_gen==0))}/{sum(y_gen==0)})")
    print(f"  ROC-AUC:           {res_gen['roc_auc']:.4f}")
    
    # 3. External Transfer Validation on 13 Independent TPEHGT Records
    df_tpehgt = pd.read_csv(os.path.join(data_dir, 'features_69_self_normalized_tpehgt13.csv'))
    df_tpehgt_rectime = parse_tpehgt_rectime(tpehgt_root)
    df_gt_merged = pd.merge(df_tpehgt, df_tpehgt_rectime, on='record_id')
    
    X_gt = df_gt_merged[feature_names].values
    y_gt = df_gt_merged['label'].values
    
    # Train full models on TPEHGDB datasets
    scaler_late = StandardScaler()
    X_late_sc = scaler_late.fit_transform(X_late)
    model_late = build_soft_ensemble(random_state=42)
    model_late.fit(X_late_sc, y_late)
    
    scaler_gen = StandardScaler()
    X_gen_sc = scaler_gen.fit_transform(X_gen)
    model_gen = build_soft_ensemble(random_state=42)
    model_gen.fit(X_gen_sc, y_gen)
    
    # Test on external cohort
    X_gt_sc_gen = scaler_gen.transform(X_gt)
    preds_gt_gen = model_gen.predict(X_gt_sc_gen)
    probs_gt_gen = model_gen.predict_proba(X_gt_sc_gen)[:, 1]
    
    acc_gt = accuracy_score(y_gt, preds_gt_gen)
    bal_gt = balanced_accuracy_score(y_gt, preds_gt_gen)
    sens_gt = recall_score(y_gt, preds_gt_gen, pos_label=1)
    spec_gt = recall_score(y_gt, preds_gt_gen, pos_label=0)
    auc_gt = roc_auc_score(y_gt, probs_gt_gen)
    
    print("\n=================== EXTERNAL TRANSFER (TPEHGT N=13) ===================")
    print(f"General Ensemble Transfer:")
    print(f"  Accuracy:          {acc_gt*100:.2f}%")
    print(f"  Balanced Accuracy: {bal_gt*100:.2f}%")
    print(f"  Preterm Recall:    {sens_gt*100:.2f}% ({int(sens_gt*sum(y_gt==1))}/{sum(y_gt==1)} preterms caught)")
    print(f"  Term Specificity:  {spec_gt*100:.2f}% ({int(spec_gt*sum(y_gt==0))}/{sum(y_gt==0)} terms correct)")
    print(f"  ROC-AUC:           {auc_gt:.4f}")
    
    # 4. Serialize Model Artifacts
    joblib.dump(model_late, os.path.join(models_dir, 'ensemble_late.joblib'))
    joblib.dump(scaler_late, os.path.join(models_dir, 'fitted_scaler_late.joblib'))
    
    joblib.dump(model_gen, os.path.join(models_dir, 'ensemble_general.joblib'))
    joblib.dump(scaler_gen, os.path.join(models_dir, 'fitted_scaler_general.joblib'))
    
    # Update default production models to point to general soft ensemble
    joblib.dump(model_gen, os.path.join(models_dir, 'best_triage_model.joblib'))
    joblib.dump(scaler_gen, os.path.join(models_dir, 'fitted_scaler.joblib'))
    
    with open(os.path.join(models_dir, 'feature_names.json'), 'w') as f:
        json.dump(feature_names, f, indent=2)
        
    # Save benchmark log
    report_data = [
        {
            'Tier': 'Tier 1: Late-Gestation Labor Triage (>= 26 wks)',
            'Cohort_Size': len(df_late),
            'Preterm_N': int(sum(y_late==1)),
            'Term_N': int(sum(y_late==0)),
            'Accuracy': res_late['accuracy'],
            'Balanced_Accuracy': res_late['balanced_accuracy'],
            'Sensitivity': res_late['sensitivity'],
            'Specificity': res_late['specificity'],
            'Precision': res_late['precision'],
            'F1_Score': res_late['f1'],
            'ROC_AUC': res_late['roc_auc']
        },
        {
            'Tier': 'Tier 2: General Universal Screening (All Records)',
            'Cohort_Size': len(df_general),
            'Preterm_N': int(sum(y_gen==1)),
            'Term_N': int(sum(y_gen==0)),
            'Accuracy': res_gen['accuracy'],
            'Balanced_Accuracy': res_gen['balanced_accuracy'],
            'Sensitivity': res_gen['sensitivity'],
            'Specificity': res_gen['specificity'],
            'Precision': res_gen['precision'],
            'F1_Score': res_gen['f1'],
            'ROC_AUC': res_gen['roc_auc']
        },
        {
            'Tier': 'External Transfer: General Ensemble on TPEHGT',
            'Cohort_Size': len(df_gt_merged),
            'Preterm_N': int(sum(y_gt==1)),
            'Term_N': int(sum(y_gt==0)),
            'Accuracy': acc_gt,
            'Balanced_Accuracy': bal_gt,
            'Sensitivity': sens_gt,
            'Specificity': spec_gt,
            'Precision': precision_score(y_gt, preds_gt_gen, zero_division=0),
            'F1_Score': f1_score(y_gt, preds_gt_gen, zero_division=0),
            'ROC_AUC': auc_gt
        }
    ]
    pd.DataFrame(report_data).to_csv(os.path.join(reports_dir, 'gestational_ensemble_evaluation.csv'), index=False)
    print(f"\nAll models and evaluation reports successfully saved to models/ and reports/!")


if __name__ == '__main__':
    main()
