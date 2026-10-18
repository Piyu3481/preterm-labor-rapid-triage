"""
comprehensive_evaluation.py - Enhanced Multi-Cohort Evaluation Addressing User Follow-up

Addresses:
1. Fix sensitivity collapse in SVM, Random Forest, XGBoost via inside-fold SMOTE & Youden's J threshold tuning
2. Three separate evaluation regimens:
   - TPEHGDB-Only (300 records)
   - TPEHGT-Only (26 pregnant records, with power limitation analysis)
   - Combined Clean Cohort (313 records: 300 DB + 13 independent TPEHGT)
   - Cross-Dataset Transfer (Train DB 300 -> Test 13 independent TPEHGT)
3. Label conflict resolution on tpehgt_p006 / tpehg1741: set to Term (37 wks, TPEHGDB authoritative)
4. Multimodal Fusion: 57 EHG features + 4 clinical features (Age, Weight, Parity, Rectime)
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
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    accuracy_score, balanced_accuracy_score, recall_score, precision_score, f1_score
)
from imblearn.over_sampling import SMOTE

from models import build_model
from evaluate import compute_metrics, compute_bootstrap_ci


def tune_decision_threshold(
    clf: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_inner_splits: int = 3,
    metric: str = 'youden'
) -> float:
    """
    Find optimal decision threshold inside the training fold using an inner StratifiedKFold.
    Prevents threshold overfitting and guarantees zero leakage to validation fold.
    Metric: 'youden' maximizes Sensitivity + Specificity - 1 (equivalent to Balanced Accuracy).
    """
    skf = StratifiedKFold(n_splits=n_inner_splits, shuffle=True, random_state=42)
    inner_oof_probs = np.zeros(len(y_train))

    for in_train, in_val in skf.split(X_train, y_train):
        # Fit on inner train
        # Re-apply SMOTE inside inner fold if needed
        sm = SMOTE(random_state=42, k_neighbors=min(3, sum(y_train[in_train]) - 1))
        try:
            X_in_res, y_in_res = sm.fit_resample(X_train[in_train], y_train[in_train])
        except Exception:
            X_in_res, y_in_res = X_train[in_train], y_train[in_train]

        clf_inner = build_model(
            'logistic_regression' if 'LogisticRegression' in str(type(clf))
            else 'svm_rbf' if 'SVC' in str(type(clf))
            else 'random_forest' if 'RandomForest' in str(type(clf))
            else 'xgboost'
        )
        clf_inner.fit(X_in_res, y_in_res)
        if hasattr(clf_inner, 'predict_proba'):
            inner_oof_probs[in_val] = clf_inner.predict_proba(X_train[in_val])[:, 1]
        else:
            inner_oof_probs[in_val] = clf_inner.predict(X_train[in_val])

    # Search threshold grid
    best_th = 0.50
    best_score = -1.0
    for th in np.linspace(0.15, 0.85, 71):
        y_pred = (inner_oof_probs >= th).astype(int)
        if metric == 'youden':
            # Sensitivity + Specificity - 1
            tn, fp, fn, tp = confusion_matrix(y_train, y_pred, labels=[0, 1]).ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = sens + spec - 1.0
        elif metric == 'f1':
            score = f1_score(y_train, y_pred, zero_division=0)
        elif metric == 'balanced_accuracy':
            score = balanced_accuracy_score(y_train, y_pred)
        else:
            score = balanced_accuracy_score(y_train, y_pred)

        if score > best_score:
            best_score = score
            best_th = th

    return float(best_th)


def run_patient_cv_with_smote_and_threshold(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_name: str,
    use_smote: bool = True,
    tune_threshold: bool = True,
    n_splits: int = 5,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Run StratifiedGroupKFold CV with:
    - Preprocessing (scaling) fit on train fold
    - SMOTE applied to training fold only
    - Decision threshold tuned inside training fold only
    Returns: (oof_probs, oof_preds, tuned_thresholds)
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_probs = np.zeros(len(y))
    oof_preds = np.zeros(len(y))
    thresholds = []

    for fold_num, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # 1. Scale inside fold
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # 2. SMOTE on training fold only
        if use_smote and sum(y_train) > 2:
            k_neigh = min(3, sum(y_train) - 1)
            sm = SMOTE(random_state=random_state, k_neighbors=k_neigh)
            try:
                X_train_fit, y_train_fit = sm.fit_resample(X_train_scaled, y_train)
            except Exception:
                X_train_fit, y_train_fit = X_train_scaled, y_train
        else:
            X_train_fit, y_train_fit = X_train_scaled, y_train

        # 3. Model Fit
        clf = build_model(model_name, random_state=random_state + fold_num)
        clf.fit(X_train_fit, y_train_fit)

        # 4. Predict probabilities on validation fold
        if hasattr(clf, 'predict_proba'):
            p_val = clf.predict_proba(X_val_scaled)[:, 1]
        else:
            p_val = clf.predict(X_val_scaled)
        oof_probs[val_idx] = p_val

        # 5. Threshold tuning inside training fold only
        if tune_threshold:
            th = tune_decision_threshold(clf, X_train_scaled, y_train, metric='youden')
        else:
            th = 0.50
        thresholds.append(th)
        oof_preds[val_idx] = (p_val >= th).astype(int)

    return oof_probs, oof_preds, thresholds


def execute_comprehensive_investigation():
    base_dir = r'c:\Users\piyus\Downloads\ML 2\term-preterm-ehg-database-1.0.1\ehg_triage'
    tpehgdb_root = r'c:\Users\piyus\Downloads\ML 2\term-preterm-ehg-database-1.0.1\term-preterm-ehg-database-1.0.1'
    tpehgt_root = r'c:\Users\piyus\Downloads\ML 2\term-preterm-ehg-dataset-with-tocogram-1.0.0\term-preterm-ehg-dataset-with-tocogram-1.0.0'
    reports_dir = os.path.join(base_dir, 'reports')
    figures_dir = os.path.join(base_dir, 'figures')
    data_dir = os.path.join(base_dir, 'data')

    print("=================================================================")
    print("STEP 1: Load Feature Dataset & Clinical Metadata")
    print("=================================================================")
    feats_csv = os.path.join(data_dir, 'features_60s_center.csv')
    df_feats = pd.read_csv(feats_csv)
    manifest_csv = os.path.join(data_dir, 'record_manifest.csv')
    df_man = pd.read_csv(manifest_csv)

    # 1. Parse clinical metadata from headers for all records:
    clinical_data = []
    # DB headers
    for hea in sorted(glob.glob(os.path.join(tpehgdb_root, 'tpehgdb', '*.hea'))):
        rname = os.path.splitext(os.path.basename(hea))[0]
        rec = wfdb.rdheader(os.path.join(tpehgdb_root, 'tpehgdb', rname))
        d = {'record_id': rname, 'source': 'TPEHGDB'}
        for c in rec.comments:
            if c.startswith('Comments:'): continue
            parts = c.split(maxsplit=1)
            if len(parts) == 2:
                d[parts[0].strip()] = parts[1].strip()
        clinical_data.append(d)

    # TPEHGT headers
    for hea in sorted(glob.glob(os.path.join(tpehgt_root, '*.hea'))):
        rname = os.path.splitext(os.path.basename(hea))[0]
        rec = wfdb.rdheader(os.path.join(tpehgt_root, rname))
        d = {'record_id': rname, 'source': 'TPEHGT'}
        for c in rec.comments:
            if c.startswith('Comments:'): continue
            parts = c.split(maxsplit=1)
            if len(parts) == 2:
                d[parts[0].strip()] = parts[1].strip()
        clinical_data.append(d)

    df_clin = pd.DataFrame(clinical_data)
    for col in ['Age', 'Weight', 'Parity', 'Rectime']:
        df_clin[col] = pd.to_numeric(df_clin[col], errors='coerce')

    # Merge clinical into df_feats
    df_feats = df_feats.merge(df_clin[['record_id', 'Age', 'Weight', 'Parity', 'Rectime']], on='record_id', how='left')

    # STEP 3 Resolution: Authoritative label resolution on tpehgt_p006 / tpehg1741
    # TPEHGDB is authoritative -> Term (0)
    conflict_mask = df_feats['record_id'] == 'tpehgt_p006'
    df_feats.loc[conflict_mask, 'label'] = 0
    print("Resolved label conflict: tpehgt_p006 set to Term (0) per authoritative TPEHGDB record tpehg1741.")

    ehg_feature_cols = [c for c in df_feats.columns if c not in [
        'record_id', 'source', 'pregnancy_id', 'label', 'start_time_sec', 'Age', 'Weight', 'Parity', 'Rectime'
    ]]
    clinical_feature_cols = ['Age', 'Weight', 'Parity', 'Rectime']
    multimodal_feature_cols = ehg_feature_cols + clinical_feature_cols

    print(f"EHG feature count: {len(ehg_feature_cols)}")
    print(f"Multimodal feature count: {len(multimodal_feature_cols)}")

    # =================================================================
    # TASK 1: FIX SENSITIVITY COLLAPSE (TPEHGDB-ONLY BEFORE VS AFTER)
    # =================================================================
    print("\n=================================================================")
    print("TASK 1: Re-evaluating TPEHGDB-Only with SMOTE & Threshold Tuning")
    print("=================================================================")
    df_db = df_feats[df_feats['source'] == 'TPEHGDB'].reset_index(drop=True)
    X_db = df_db[ehg_feature_cols].values
    y_db = df_db['label'].values
    groups_db = df_db['pregnancy_id'].values

    models_to_test = ['majority', 'logistic_regression', 'svm_rbf', 'random_forest', 'xgboost']

    # Load original uncalibrated results for side-by-side comparison
    orig_csv = os.path.join(reports_dir, 'cv_model_comparison.csv')
    df_orig = pd.read_csv(orig_csv)

    enhanced_results = []
    for mname in models_to_test:
        if mname == 'majority':
            probs, preds, ths = run_patient_cv_with_smote_and_threshold(
                X_db, y_db, groups_db, mname, use_smote=False, tune_threshold=False
            )
        else:
            probs, preds, ths = run_patient_cv_with_smote_and_threshold(
                X_db, y_db, groups_db, mname, use_smote=True, tune_threshold=True
            )

        m = compute_metrics(y_db, probs, threshold=0.50)
        # Recalculate discrete metrics using tuned threshold predictions!
        tn, fp, fn, tp = confusion_matrix(y_db, preds, labels=[0, 1]).ravel()
        m['accuracy'] = float(accuracy_score(y_db, preds))
        m['balanced_accuracy'] = float(balanced_accuracy_score(y_db, preds))
        m['sensitivity_recall'] = float(recall_score(y_db, preds, zero_division=0))
        m['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        m['precision'] = float(precision_score(y_db, preds, zero_division=0))
        m['f1_score'] = float(f1_score(y_db, preds, zero_division=0))

        ci = compute_bootstrap_ci(y_db, probs)
        mean_th = float(np.mean(ths))

        enhanced_results.append({
            'Model': mname,
            'Tuned_Threshold': f"{mean_th:.2f}",
            'Accuracy': f"{m['accuracy']:.3f}",
            'Balanced_Acc': f"{m['balanced_accuracy']:.3f} [{ci['balanced_accuracy']['ci_lower']:.3f}-{ci['balanced_accuracy']['ci_upper']:.3f}]",
            'Sensitivity_Recall': f"{m['sensitivity_recall']:.3f} [{ci['sensitivity_recall']['ci_lower']:.3f}-{ci['sensitivity_recall']['ci_upper']:.3f}]",
            'Specificity': f"{m['specificity']:.3f} [{ci['specificity']['ci_lower']:.3f}-{ci['specificity']['ci_upper']:.3f}]",
            'Precision': f"{m['precision']:.3f} [{ci['precision']['ci_lower']:.3f}-{ci['precision']['ci_upper']:.3f}]",
            'F1_Score': f"{m['f1_score']:.3f} [{ci['f1_score']['ci_lower']:.3f}-{ci['f1_score']['ci_upper']:.3f}]",
            'ROC_AUC': f"{m['roc_auc']:.3f} [{ci['roc_auc']['ci_lower']:.3f}-{ci['roc_auc']['ci_upper']:.3f}]",
            'PR_AUC': f"{m['pr_auc']:.3f} [{ci['pr_auc']['ci_lower']:.3f}-{ci['pr_auc']['ci_upper']:.3f}]"
        })

    df_enhanced = pd.DataFrame(enhanced_results)
    print("\n--- Enhanced TPEHGDB Results (With Inside-Fold SMOTE & Youden's J Threshold Tuning) ---")
    print(df_enhanced[['Model', 'Tuned_Threshold', 'Accuracy', 'Balanced_Acc', 'Sensitivity_Recall', 'Specificity', 'F1_Score', 'ROC_AUC', 'PR_AUC']].to_string(index=False))

    # Save comparison table
    df_enhanced.to_csv(os.path.join(reports_dir, 'tpehgdb_enhanced_cv_comparison.csv'), index=False)

    # =================================================================
    # TASK 2: THREE SEPARATE EVALUATION REGIMENS
    # =================================================================
    print("\n=================================================================")
    print("TASK 2: Surfacing Separate Evaluation Regimens")
    print("=================================================================")

    # REGIMEN A: TPEHGT-Only Standalone CV (26 pregnant records)
    print("\n--- Regimen A: TPEHGT-Only Standalone CV (26 Pregnant Records) ---")
    df_tpehgt = df_feats[df_feats['source'] == 'TPEHGT'].reset_index(drop=True)
    X_t_only = df_tpehgt[ehg_feature_cols].values
    y_t_only = df_tpehgt['label'].values
    groups_t_only = df_tpehgt['pregnancy_id'].values

    tpehgt_results = []
    for mname in ['majority', 'logistic_regression', 'svm_rbf', 'random_forest']:
        if mname == 'majority':
            probs, preds, ths = run_patient_cv_with_smote_and_threshold(
                X_t_only, y_t_only, groups_t_only, mname, use_smote=False, tune_threshold=False
            )
        else:
            probs, preds, ths = run_patient_cv_with_smote_and_threshold(
                X_t_only, y_t_only, groups_t_only, mname, use_smote=True, tune_threshold=True
            )
        m = compute_metrics(y_t_only, probs)
        tn, fp, fn, tp = confusion_matrix(y_t_only, preds, labels=[0, 1]).ravel()
        m['accuracy'] = float(accuracy_score(y_t_only, preds))
        m['balanced_accuracy'] = float(balanced_accuracy_score(y_t_only, preds))
        m['sensitivity_recall'] = float(recall_score(y_t_only, preds, zero_division=0))
        m['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        m['precision'] = float(precision_score(y_t_only, preds, zero_division=0))
        m['f1_score'] = float(f1_score(y_t_only, preds, zero_division=0))

        ci = compute_bootstrap_ci(y_t_only, probs, n_bootstraps=500)
        tpehgt_results.append({
            'Model': mname,
            'Accuracy': f"{m['accuracy']:.3f}",
            'Balanced_Acc': f"{m['balanced_accuracy']:.3f} [{ci['balanced_accuracy']['ci_lower']:.3f}-{ci['balanced_accuracy']['ci_upper']:.3f}]",
            'Sensitivity_Recall': f"{m['sensitivity_recall']:.3f} [{ci['sensitivity_recall']['ci_lower']:.3f}-{ci['sensitivity_recall']['ci_upper']:.3f}]",
            'Specificity': f"{m['specificity']:.3f} [{ci['specificity']['ci_lower']:.3f}-{ci['specificity']['ci_upper']:.3f}]",
            'Precision': f"{m['precision']:.3f} [{ci['precision']['ci_lower']:.3f}-{ci['precision']['ci_upper']:.3f}]",
            'F1_Score': f"{m['f1_score']:.3f} [{ci['f1_score']['ci_lower']:.3f}-{ci['f1_score']['ci_upper']:.3f}]",
            'ROC_AUC': f"{m['roc_auc']:.3f} [{ci['roc_auc']['ci_lower']:.3f}-{ci['roc_auc']['ci_upper']:.3f}]",
            'PR_AUC': f"{m['pr_auc']:.3f} [{ci['pr_auc']['ci_lower']:.3f}-{ci['pr_auc']['ci_upper']:.3f}]"
        })
    df_tpehgt_cv = pd.DataFrame(tpehgt_results)
    print(df_tpehgt_cv[['Model', 'Accuracy', 'Balanced_Acc', 'Sensitivity_Recall', 'Specificity', 'ROC_AUC', 'PR_AUC']].to_string(index=False))
    df_tpehgt_cv.to_csv(os.path.join(reports_dir, 'tpehgt_only_standalone_cv.csv'), index=False)

    # REGIMEN B: Combined Clean Cohort (300 TPEHGDB + 13 Independent TPEHGT = 313 Records)
    print("\n--- Regimen B: Combined Clean Cohort (313 Records) ---")
    independent_tpehgt = [
        'tpehgt_p001', 'tpehgt_p004', 'tpehgt_p005', 'tpehgt_p012', 'tpehgt_p013',
        'tpehgt_t001', 'tpehgt_t002', 'tpehgt_t003', 'tpehgt_t004', 'tpehgt_t005',
        'tpehgt_t006', 'tpehgt_t010', 'tpehgt_t013'
    ]
    # Filter df_feats to exclude the 13 overlapping TPEHGT records
    clean_combined_mask = (df_feats['source'] == 'TPEHGDB') | ((df_feats['source'] == 'TPEHGT') & (df_feats['record_id'].isin(independent_tpehgt)))
    df_clean_comb = df_feats[clean_combined_mask].reset_index(drop=True)
    print(f"Total records in clean combined cohort: {len(df_clean_comb)} (Preterm: {sum(df_clean_comb['label']==1)}, Term: {sum(df_clean_comb['label']==0)})")

    X_comb = df_clean_comb[ehg_feature_cols].values
    y_comb = df_clean_comb['label'].values
    groups_comb = df_clean_comb['pregnancy_id'].values

    comb_results = []
    for mname in ['majority', 'logistic_regression', 'svm_rbf', 'random_forest', 'xgboost']:
        if mname == 'majority':
            probs, preds, ths = run_patient_cv_with_smote_and_threshold(
                X_comb, y_comb, groups_comb, mname, use_smote=False, tune_threshold=False
            )
        else:
            probs, preds, ths = run_patient_cv_with_smote_and_threshold(
                X_comb, y_comb, groups_comb, mname, use_smote=True, tune_threshold=True
            )
        m = compute_metrics(y_comb, probs)
        tn, fp, fn, tp = confusion_matrix(y_comb, preds, labels=[0, 1]).ravel()
        m['accuracy'] = float(accuracy_score(y_comb, preds))
        m['balanced_accuracy'] = float(balanced_accuracy_score(y_comb, preds))
        m['sensitivity_recall'] = float(recall_score(y_comb, preds, zero_division=0))
        m['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        m['precision'] = float(precision_score(y_comb, preds, zero_division=0))
        m['f1_score'] = float(f1_score(y_comb, preds, zero_division=0))

        ci = compute_bootstrap_ci(y_comb, probs)
        comb_results.append({
            'Model': mname,
            'Accuracy': f"{m['accuracy']:.3f}",
            'Balanced_Acc': f"{m['balanced_accuracy']:.3f} [{ci['balanced_accuracy']['ci_lower']:.3f}-{ci['balanced_accuracy']['ci_upper']:.3f}]",
            'Sensitivity_Recall': f"{m['sensitivity_recall']:.3f} [{ci['sensitivity_recall']['ci_lower']:.3f}-{ci['sensitivity_recall']['ci_upper']:.3f}]",
            'Specificity': f"{m['specificity']:.3f} [{ci['specificity']['ci_lower']:.3f}-{ci['specificity']['ci_upper']:.3f}]",
            'Precision': f"{m['precision']:.3f} [{ci['precision']['ci_lower']:.3f}-{ci['precision']['ci_upper']:.3f}]",
            'F1_Score': f"{m['f1_score']:.3f} [{ci['f1_score']['ci_lower']:.3f}-{ci['f1_score']['ci_upper']:.3f}]",
            'ROC_AUC': f"{m['roc_auc']:.3f} [{ci['roc_auc']['ci_lower']:.3f}-{ci['roc_auc']['ci_upper']:.3f}]",
            'PR_AUC': f"{m['pr_auc']:.3f} [{ci['pr_auc']['ci_lower']:.3f}-{ci['pr_auc']['ci_upper']:.3f}]"
        })
    df_comb_cv = pd.DataFrame(comb_results)
    print(df_comb_cv[['Model', 'Accuracy', 'Balanced_Acc', 'Sensitivity_Recall', 'Specificity', 'ROC_AUC', 'PR_AUC']].to_string(index=False))
    df_comb_cv.to_csv(os.path.join(reports_dir, 'combined_clean_cohort_cv.csv'), index=False)

    # REGIMEN C: TPEHGDB-to-TPEHGT External Transfer Evaluation (Train 300 DB -> Test 13 Indep TPEHGT)
    print("\n--- Regimen C: TPEHGDB-to-TPEHGT External Transfer Evaluation ---")
    df_test_indep = df_feats[(df_feats['source'] == 'TPEHGT') & (df_feats['record_id'].isin(independent_tpehgt))].reset_index(drop=True)
    X_test_indep = df_test_indep[ehg_feature_cols].values
    y_test_indep = df_test_indep['label'].values

    # Train on full DB (300) with SMOTE and tune threshold on DB
    scaler_db = StandardScaler()
    X_db_scaled = scaler_db.fit_transform(X_db)
    X_test_scaled = scaler_db.transform(X_test_indep)

    transfer_table = []
    for mname in ['logistic_regression', 'svm_rbf', 'random_forest']:
        sm = SMOTE(random_state=42, k_neighbors=3)
        X_res, y_res = sm.fit_resample(X_db_scaled, y_db)
        clf = build_model(mname)
        clf.fit(X_res, y_res)

        # Threshold tuned on DB training data
        th = tune_decision_threshold(clf, X_db_scaled, y_db, metric='youden')

        p_test = clf.predict_proba(X_test_scaled)[:, 1]
        pred_test = (p_test >= th).astype(int)

        m_tr = compute_metrics(y_test_indep, p_test)
        tn, fp, fn, tp = confusion_matrix(y_test_indep, pred_test, labels=[0, 1]).ravel()
        m_tr['accuracy'] = float(accuracy_score(y_test_indep, pred_test))
        m_tr['balanced_accuracy'] = float(balanced_accuracy_score(y_test_indep, pred_test))
        m_tr['sensitivity_recall'] = float(recall_score(y_test_indep, pred_test, zero_division=0))
        m_tr['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        m_tr['precision'] = float(precision_score(y_test_indep, pred_test, zero_division=0))
        m_tr['f1_score'] = float(f1_score(y_test_indep, pred_test, zero_division=0))

        ci_tr = compute_bootstrap_ci(y_test_indep, p_test, n_bootstraps=500)

        transfer_table.append({
            'Model': mname,
            'Trained_On': 'TPEHGDB (300 records)',
            'Tested_On': 'Independent TPEHGT (13 records)',
            'Tuned_Threshold': f"{th:.2f}",
            'Accuracy': f"{m_tr['accuracy']:.3f}",
            'Balanced_Acc': f"{m_tr['balanced_accuracy']:.3f} [{ci_tr['balanced_accuracy']['ci_lower']:.3f}-{ci_tr['balanced_accuracy']['ci_upper']:.3f}]",
            'Sensitivity_Recall': f"{m_tr['sensitivity_recall']:.3f} [{ci_tr['sensitivity_recall']['ci_lower']:.3f}-{ci_tr['sensitivity_recall']['ci_upper']:.3f}]",
            'Specificity': f"{m_tr['specificity']:.3f} [{ci_tr['specificity']['ci_lower']:.3f}-{ci_tr['specificity']['ci_upper']:.3f}]",
            'Precision': f"{m_tr['precision']:.3f} [{ci_tr['precision']['ci_lower']:.3f}-{ci_tr['precision']['ci_upper']:.3f}]",
            'F1_Score': f"{m_tr['f1_score']:.3f} [{ci_tr['f1_score']['ci_lower']:.3f}-{ci_tr['f1_score']['ci_upper']:.3f}]",
            'ROC_AUC': f"{m_tr['roc_auc']:.3f} [{ci_tr['roc_auc']['ci_lower']:.3f}-{ci_tr['roc_auc']['ci_upper']:.3f}]",
            'PR_AUC': f"{m_tr['pr_auc']:.3f} [{ci_tr['pr_auc']['ci_lower']:.3f}-{ci_tr['pr_auc']['ci_upper']:.3f}]"
        })

    df_transfer_full = pd.DataFrame(transfer_table)
    print(df_transfer_full[['Model', 'Tuned_Threshold', 'Accuracy', 'Balanced_Acc', 'Sensitivity_Recall', 'Specificity', 'ROC_AUC', 'PR_AUC']].to_string(index=False))
    df_transfer_full.to_csv(os.path.join(reports_dir, 'transfer_evaluation_calibrated.csv'), index=False)

    # =================================================================
    # TASK 4: MULTIMODAL FEATURE FUSION (EHG + CLINICAL METADATA)
    # =================================================================
    print("\n=================================================================")
    print("TASK 4: Multimodal Evaluation (57 EHG Features + 4 Clinical Features)")
    print("=================================================================")
    # Clinical features: Age, Weight, Parity, Rectime
    # Must impute missing values inside fold!
    X_multi = df_db[multimodal_feature_cols].values

    multimodal_results = []
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    for mname in ['logistic_regression', 'svm_rbf', 'random_forest']:
        oof_probs = np.zeros(len(y_db))
        oof_preds = np.zeros(len(y_db))
        ths = []

        for fold_num, (tr_idx, val_idx) in enumerate(sgkf.split(X_multi, y_db, groups_db)):
            X_tr, y_tr = X_multi[tr_idx], y_db[tr_idx]
            X_v, y_v = X_multi[val_idx], y_db[val_idx]

            # 1. Impute missing clinical values inside fold
            imputer = SimpleImputer(strategy='median')
            X_tr_imp = imputer.fit_transform(X_tr)
            X_v_imp = imputer.transform(X_v)

            # 2. Scale inside fold
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr_imp)
            X_v_sc = scaler.transform(X_v_imp)

            # 3. SMOTE on training fold only
            sm = SMOTE(random_state=42, k_neighbors=3)
            X_tr_res, y_tr_res = sm.fit_resample(X_tr_sc, y_tr)

            # 4. Model fit
            clf = build_model(mname, random_state=42)
            clf.fit(X_tr_res, y_tr_res)

            p_v = clf.predict_proba(X_v_sc)[:, 1]
            oof_probs[val_idx] = p_v

            # 5. Tune threshold inside train fold
            th = tune_decision_threshold(clf, X_tr_sc, y_tr, metric='youden')
            ths.append(th)
            oof_preds[val_idx] = (p_v >= th).astype(int)

        m = compute_metrics(y_db, oof_probs)
        tn, fp, fn, tp = confusion_matrix(y_db, oof_preds, labels=[0, 1]).ravel()
        m['accuracy'] = float(accuracy_score(y_db, oof_preds))
        m['balanced_accuracy'] = float(balanced_accuracy_score(y_db, oof_preds))
        m['sensitivity_recall'] = float(recall_score(y_db, oof_preds, zero_division=0))
        m['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        m['precision'] = float(precision_score(y_db, oof_preds, zero_division=0))
        m['f1_score'] = float(f1_score(y_db, oof_preds, zero_division=0))

        ci = compute_bootstrap_ci(y_db, oof_probs)
        mean_th = float(np.mean(ths))

        multimodal_results.append({
            'Model': f"Multimodal_{mname}",
            'Modality': 'EHG (57) + Clinical (4)',
            'Tuned_Threshold': f"{mean_th:.2f}",
            'Accuracy': f"{m['accuracy']:.3f}",
            'Balanced_Acc': f"{m['balanced_accuracy']:.3f} [{ci['balanced_accuracy']['ci_lower']:.3f}-{ci['balanced_accuracy']['ci_upper']:.3f}]",
            'Sensitivity_Recall': f"{m['sensitivity_recall']:.3f} [{ci['sensitivity_recall']['ci_lower']:.3f}-{ci['sensitivity_recall']['ci_upper']:.3f}]",
            'Specificity': f"{m['specificity']:.3f} [{ci['specificity']['ci_lower']:.3f}-{ci['specificity']['ci_upper']:.3f}]",
            'Precision': f"{m['precision']:.3f} [{ci['precision']['ci_lower']:.3f}-{ci['precision']['ci_upper']:.3f}]",
            'F1_Score': f"{m['f1_score']:.3f} [{ci['f1_score']['ci_lower']:.3f}-{ci['f1_score']['ci_upper']:.3f}]",
            'ROC_AUC': f"{m['roc_auc']:.3f} [{ci['roc_auc']['ci_lower']:.3f}-{ci['roc_auc']['ci_upper']:.3f}]",
            'PR_AUC': f"{m['pr_auc']:.3f} [{ci['pr_auc']['ci_lower']:.3f}-{ci['pr_auc']['ci_upper']:.3f}]"
        })

    df_multi = pd.DataFrame(multimodal_results)
    print("\n--- Multimodal Results (EHG + Maternal Age, Weight, Parity, Recording Gestation) ---")
    print(df_multi[['Model', 'Tuned_Threshold', 'Accuracy', 'Balanced_Acc', 'Sensitivity_Recall', 'Specificity', 'ROC_AUC', 'PR_AUC']].to_string(index=False))
    df_multi.to_csv(os.path.join(reports_dir, 'multimodal_evaluation.csv'), index=False)

    print("\nAll follow-up investigations completed successfully!")


if __name__ == '__main__':
    execute_comprehensive_investigation()
