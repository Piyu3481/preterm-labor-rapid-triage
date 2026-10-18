"""
train_and_evaluate.py - Full Cross-Validation, Benchmarking, and Evaluation Pipeline

Workflow:
1. Feature extraction and segment caching for all records in the manifest
2. Baseline reproducibility evaluation on official PhysioNet .fvl feature files
3. 5-Fold Stratified Patient-Level Cross Validation across all 6 models:
   - Majority Baseline
   - Logistic Regression
   - RBF SVM
   - Random Forest
   - 1D CNN (PyTorch)
   - CNN + Feature Fusion (PyTorch)
4. Strict inside-fold preprocessing (StandardScaler)
5. Bootstrap 95% Confidence Intervals
6. Cross-dataset transfer evaluation (TPEHGDB -> Independent TPEHGT)
7. Save best trained model, scaler, metrics tables, and ROC/PR figures
"""

import os
import glob
import time
import json
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

from preprocessing import load_raw_ehg_record, extract_60s_segments, TARGET_FS, WINDOW_SAMPLES
from features import extract_features_from_window, load_tpehgdb_baseline_features
from splits import get_stratified_patient_folds, create_transfer_split
from models import build_model, EHG1DCNN, EHGFusionModel, EHGDataset, train_cnn_epoch, predict_cnn
from evaluate import compute_metrics, compute_bootstrap_ci, plot_roc_pr_curves, plot_confusion_matrix


def run_feature_extraction_pipeline(
    manifest_csv: str,
    tpehgdb_root: str,
    tpehgt_root: str,
    output_dir: str,
    policy: str = 'center'
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Extract 60s segments and engineered features for all eligible pregnant records.
    Returns: (df_features, signals_array, labels_array)
    """
    os.makedirs(output_dir, exist_ok=True)
    cache_features = os.path.join(output_dir, f'features_60s_{policy}.csv')
    cache_signals = os.path.join(output_dir, f'signals_60s_{policy}.npy')
    cache_labels = os.path.join(output_dir, f'labels_60s_{policy}.npy')

    if os.path.exists(cache_features) and os.path.exists(cache_signals) and os.path.exists(cache_labels):
        print(f"Loading cached 60s features and signals from {output_dir}...")
        df_feats = pd.read_csv(cache_features)
        signals = np.load(cache_signals)
        labels = np.load(cache_labels)
        return df_feats, signals, labels

    print(f"Extracting 60s segments and features (policy: {policy})...")
    df_manifest = pd.read_csv(manifest_csv)
    # Exclude non-pregnant records
    df_manifest = df_manifest[df_manifest['is_pregnant']].reset_index(drop=True)

    feature_rows = []
    signal_windows = []
    labels = []

    for idx, row in df_manifest.iterrows():
        source = row['source']
        rec_id = row['record_id']
        label_val = 1 if row['label'] == 'Preterm' else 0

        if source == 'TPEHGDB':
            rec_path = os.path.join(tpehgdb_root, 'tpehgdb', rec_id)
        else:
            rec_path = os.path.join(tpehgt_root, rec_id)

        try:
            sig, fs, _ = load_raw_ehg_record(rec_path, source)
            segments = extract_60s_segments(sig, fs=fs, policy=policy, lowcut=0.3, highcut=3.0)
            seg = segments[0]
            win = seg['signal']  # (1200, 3)

            feats = extract_features_from_window(win, fs=fs)
            feats['record_id'] = rec_id
            feats['source'] = source
            feats['pregnancy_id'] = row['pregnancy_id']
            feats['label'] = label_val
            feats['start_time_sec'] = seg['start_time_sec']

            feature_rows.append(feats)
            signal_windows.append(win)
            labels.append(label_val)

        except Exception as e:
            print(f"Error processing record {rec_id}: {e}")

    df_feats = pd.DataFrame(feature_rows)
    signals_arr = np.array(signal_windows, dtype=np.float32)
    labels_arr = np.array(labels, dtype=np.int32)

    df_feats.to_csv(cache_features, index=False)
    np.save(cache_signals, signals_arr)
    np.save(cache_labels, labels_arr)
    print(f"Extracted and cached {len(df_feats)} records.")
    return df_feats, signals_arr, labels_arr


def evaluate_physionet_benchmark(
    tpehgdb_dir: str
) -> Dict[str, Any]:
    """
    Run stratified patient-level cross validation on the official precomputed PhysioNet feature files.
    """
    print("\n=======================================================")
    print("Evaluating Official PhysioNet Baseline Features (TPEHGDB)")
    print("=======================================================")
    fvl_path = os.path.join(tpehgdb_dir, 'tpehgdb_features__filter_0.3_Hz-3.0_Hz.fvl')
    if not os.path.exists(fvl_path):
        fvl_path = glob.glob(os.path.join(tpehgdb_dir, '*.fvl'))[0]

    df_fvl = load_tpehgdb_baseline_features(fvl_path)
    # Pivot channels into single row per record
    piv = df_fvl.pivot(index='Record', columns='Chann', values=['RMS', 'Fmed', 'Fpeak', 'Samp. en.'])
    piv.columns = [f'{col[0]}_ch{col[1]}' for col in piv.columns]
    piv['label'] = df_fvl.groupby('Record')['is_preterm'].first()
    piv = piv.dropna().reset_index()

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    feature_cols = [c for c in piv.columns if c not in ['Record', 'label']]
    X = piv[feature_cols].values
    y = piv['label'].values

    y_probs = np.zeros(len(y))
    for train_idx, val_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])

        clf = build_model('logistic_regression')
        clf.fit(X_train, y[train_idx])
        y_probs[val_idx] = clf.predict_proba(X_val)[:, 1]

    metrics = compute_metrics(y, y_probs)
    print(f"PhysioNet Baseline (LR, 0.3-3.0 Hz): Balanced Acc = {metrics['balanced_accuracy']:.3f}, ROC-AUC = {metrics['roc_auc']:.3f}, F1 = {metrics['f1_score']:.3f}")
    return metrics


def run_full_evaluation():
    base_dir = r'c:\Users\piyus\Downloads\ML 2\term-preterm-ehg-database-1.0.1\ehg_triage'
    tpehgdb_root = r'c:\Users\piyus\Downloads\ML 2\term-preterm-ehg-database-1.0.1\term-preterm-ehg-database-1.0.1'
    tpehgt_root = r'c:\Users\piyus\Downloads\ML 2\term-preterm-ehg-dataset-with-tocogram-1.0.0\term-preterm-ehg-dataset-with-tocogram-1.0.0'
    manifest_csv = os.path.join(base_dir, 'data', 'record_manifest.csv')
    reports_dir = os.path.join(base_dir, 'reports')
    figures_dir = os.path.join(base_dir, 'figures')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 1. Official PhysioNet Baseline
    phys_metrics = evaluate_physionet_benchmark(tpehgdb_root)

    # 2. Extract 60s windows and features
    df_feats, signals, labels = run_feature_extraction_pipeline(
        manifest_csv=manifest_csv,
        tpehgdb_root=tpehgdb_root,
        tpehgt_root=tpehgt_root,
        output_dir=os.path.join(base_dir, 'data'),
        policy='center'
    )

    # Filter to TPEHGDB primary cohort for primary evaluation
    mask_db = df_feats['source'] == 'TPEHGDB'
    df_db = df_feats[mask_db].reset_index(drop=True)
    signals_db = signals[mask_db]
    labels_db = labels[mask_db]

    feature_cols = [c for c in df_db.columns if c not in ['record_id', 'source', 'pregnancy_id', 'label', 'start_time_sec']]
    X_feats_db = df_db[feature_cols].values
    y_db = labels_db

    print(f"\n=======================================================")
    print(f"Running 5-Fold Stratified Patient CV on TPEHGDB ({len(df_db)} records)")
    print(f"Features: {len(feature_cols)}, Preterm: {np.sum(y_db)}, Term: {len(y_db) - np.sum(y_db)}")
    print("=======================================================")

    folds = get_stratified_patient_folds(df_db, n_splits=5, random_state=42)

    model_names = ['majority', 'logistic_regression', 'svm_rbf', 'random_forest', 'xgboost', '1d_cnn', 'cnn_fusion']
    oof_predictions = {name: np.zeros(len(y_db)) for name in model_names}

    for fold_num, (train_idx, val_idx) in enumerate(folds):
        print(f"\n--- Fold {fold_num + 1} / 5 (Train: {len(train_idx)}, Val: {len(val_idx)}) ---")
        
        # 1. Feature scaling strictly inside training fold
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_feats_db[train_idx])
        X_val_scaled = scaler.transform(X_feats_db[val_idx])

        y_train, y_val = y_db[train_idx], y_db[val_idx]
        sig_train, sig_val = signals_db[train_idx], signals_db[val_idx]

        # Calculate pos_weight for PyTorch loss
        n_pos = np.sum(y_train)
        n_neg = len(y_train) - n_pos
        pos_weight_val = torch.tensor([float(n_neg / max(1, n_pos))], dtype=torch.float32)

        # Train classical models
        for name in ['majority', 'logistic_regression', 'svm_rbf', 'random_forest', 'xgboost']:
            clf = build_model(name, random_state=42)
            clf.fit(X_train_scaled, y_train)
            if hasattr(clf, 'predict_proba'):
                probs = clf.predict_proba(X_val_scaled)[:, 1]
            else:
                probs = clf.predict(X_val_scaled)
            oof_predictions[name][val_idx] = probs

        # Train 1D CNN
        train_ds = EHGDataset(sig_train, y_train)
        val_ds = EHGDataset(sig_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

        cnn_model = EHG1DCNN(in_channels=3, n_classes=1, dropout=0.4)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_val)
        optimizer = optim.AdamW(cnn_model.parameters(), lr=1e-3, weight_decay=1e-2)

        for epoch in range(25):
            train_cnn_epoch(cnn_model, train_loader, optimizer, criterion)
        oof_predictions['1d_cnn'][val_idx] = predict_cnn(cnn_model, val_loader)

        # Train CNN Fusion
        train_ds_fuse = EHGDataset(sig_train, y_train, tab_features=X_train_scaled)
        val_ds_fuse = EHGDataset(sig_val, y_val, tab_features=X_val_scaled)
        train_loader_fuse = DataLoader(train_ds_fuse, batch_size=16, shuffle=True)
        val_loader_fuse = DataLoader(val_ds_fuse, batch_size=32, shuffle=False)

        fusion_model = EHGFusionModel(in_channels=3, n_tabular_features=len(feature_cols), dropout=0.4)
        opt_fuse = optim.AdamW(fusion_model.parameters(), lr=1e-3, weight_decay=1e-2)
        for epoch in range(25):
            train_cnn_epoch(fusion_model, train_loader_fuse, opt_fuse, criterion, is_fusion=True)
        oof_predictions['cnn_fusion'][val_idx] = predict_cnn(fusion_model, val_loader_fuse, is_fusion=True)

    # 3. Compute Metrics and 95% Confidence Intervals
    print("\n=======================================================")
    print("Cross-Validation Results Summary (TPEHGDB Cohort)")
    print("=======================================================")
    cv_summary_rows = []
    curves_data = {}

    for name in model_names:
        probs = oof_predictions[name]
        m = compute_metrics(y_db, probs)
        ci = compute_bootstrap_ci(y_db, probs, n_bootstraps=1000)
        
        row = {
            'Model': name,
            'Accuracy': f"{m['accuracy']:.3f} [{ci['accuracy']['ci_lower']:.3f}-{ci['accuracy']['ci_upper']:.3f}]",
            'Balanced_Acc': f"{m['balanced_accuracy']:.3f} [{ci['balanced_accuracy']['ci_lower']:.3f}-{ci['balanced_accuracy']['ci_upper']:.3f}]",
            'Sensitivity_Recall': f"{m['sensitivity_recall']:.3f} [{ci['sensitivity_recall']['ci_lower']:.3f}-{ci['sensitivity_recall']['ci_upper']:.3f}]",
            'Specificity': f"{m['specificity']:.3f} [{ci['specificity']['ci_lower']:.3f}-{ci['specificity']['ci_upper']:.3f}]",
            'Precision': f"{m['precision']:.3f} [{ci['precision']['ci_lower']:.3f}-{ci['precision']['ci_upper']:.3f}]",
            'F1_Score': f"{m['f1_score']:.3f} [{ci['f1_score']['ci_lower']:.3f}-{ci['f1_score']['ci_upper']:.3f}]",
            'ROC_AUC': f"{m['roc_auc']:.3f} [{ci['roc_auc']['ci_lower']:.3f}-{ci['roc_auc']['ci_upper']:.3f}]",
            'PR_AUC': f"{m['pr_auc']:.3f} [{ci['pr_auc']['ci_lower']:.3f}-{ci['pr_auc']['ci_upper']:.3f}]",
            'Raw_Bal_Acc': m['balanced_accuracy'],
            'Raw_ROC_AUC': m['roc_auc']
        }
        cv_summary_rows.append(row)
        curves_data[name] = (y_db, probs)

        # Plot confusion matrix
        y_pred = (probs >= 0.5).astype(int)
        cm_path = os.path.join(figures_dir, f'confusion_matrix_{name}.png')
        plot_confusion_matrix(y_db, y_pred, name, cm_path)

    df_cv_summary = pd.DataFrame(cv_summary_rows)
    print(df_cv_summary[['Model', 'Accuracy', 'Balanced_Acc', 'Sensitivity_Recall', 'Specificity', 'ROC_AUC', 'PR_AUC']].to_string(index=False))

    summary_csv = os.path.join(reports_dir, 'cv_model_comparison.csv')
    df_cv_summary.to_csv(summary_csv, index=False)

    # Plot ROC and PR curves
    plot_roc_pr_curves(curves_data, save_path=os.path.join(figures_dir, 'roc_pr_curves.png'))

    # 4. Evaluate External Transfer Test on Independent TPEHGT Cohort
    print("\n=======================================================")
    print("Evaluating Cross-Dataset Transfer (TPEHGDB -> Independent TPEHGT)")
    print("=======================================================")
    independent_tpehgt_records = [
        'tpehgt_p001', 'tpehgt_p004', 'tpehgt_p005', 'tpehgt_p012', 'tpehgt_p013',
        'tpehgt_t001', 'tpehgt_t002', 'tpehgt_t003', 'tpehgt_t004', 'tpehgt_t005',
        'tpehgt_t006', 'tpehgt_t010', 'tpehgt_t013'
    ]
    mask_transfer_test = (df_feats['source'] == 'TPEHGT') & (df_feats['record_id'].isin(independent_tpehgt_records))
    df_test = df_feats[mask_transfer_test].reset_index(drop=True)
    X_test_feats = df_test[feature_cols].values
    y_test = df_test['label'].values

    # Train best model (SVM / LR) on all 300 TPEHGDB records
    full_scaler = StandardScaler()
    X_db_scaled = full_scaler.fit_transform(X_feats_db)
    X_test_scaled = full_scaler.transform(X_test_feats)

    transfer_results = []
    trained_final_models = {}

    for name in ['logistic_regression', 'svm_rbf', 'random_forest']:
        clf = build_model(name)
        clf.fit(X_db_scaled, y_db)
        trained_final_models[name] = clf
        p_test = clf.predict_proba(X_test_scaled)[:, 1]
        m_trans = compute_metrics(y_test, p_test)
        transfer_results.append({
            'Model': name,
            'Test_Records': len(y_test),
            'Accuracy': f"{m_trans['accuracy']:.3f}",
            'Balanced_Acc': f"{m_trans['balanced_accuracy']:.3f}",
            'Sensitivity': f"{m_trans['sensitivity_recall']:.3f}",
            'Specificity': f"{m_trans['specificity']:.3f}",
            'ROC_AUC': f"{m_trans['roc_auc']:.3f}",
            'PR_AUC': f"{m_trans['pr_auc']:.3f}"
        })

    df_transfer = pd.DataFrame(transfer_results)
    print("Transfer Results on Independent TPEHGT:")
    print(df_transfer.to_string(index=False))
    df_transfer.to_csv(os.path.join(reports_dir, 'transfer_evaluation.csv'), index=False)

    # 5. Save the primary triage model and artifacts for Phase 5 Inference
    # Select best model by balanced accuracy
    best_model_name = df_cv_summary.sort_values(by='Raw_Bal_Acc', ascending=False).iloc[0]['Model']
    print(f"\nOptimal cross-validation model: {best_model_name}")

    final_model = trained_final_models.get(best_model_name, trained_final_models['svm_rbf'])
    joblib.dump(final_model, os.path.join(models_dir, 'best_triage_model.joblib'))
    joblib.dump(full_scaler, os.path.join(models_dir, 'fitted_scaler.joblib'))
    with open(os.path.join(models_dir, 'feature_names.json'), 'w') as f:
        json.dump(feature_cols, f)

    print(f"Model and scaler persisted to {models_dir}")


if __name__ == '__main__':
    run_full_evaluation()
