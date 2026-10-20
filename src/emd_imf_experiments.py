"""
emd_imf_experiments.py - Empirical Mode Decomposition (EMD) and Intrinsic Mode Function (IMF) Benchmark.

Evaluates:
1. Extraction of 48 IMF features across leads S1, S2, S3 (Energy, Relative Energy, Median Frequency, Sample Entropy).
2. Standalone 48 IMF feature evaluation under 5-fold Stratified Patient CV.
3. Hybrid Fusion: Original 69 self-normalized features + 48 IMF features = 117 features.
4. Model comparison: ExtraTrees, Logistic Regression, Gaussian RBF SVM.
5. External validation on 13 independent TPEHGT records.
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
from PyEMD import EMD
import antropy as ant
import scipy.signal as signal
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    roc_auc_score,
    precision_score,
    f1_score
)

# Ensure src in path
sys.path.append(os.path.dirname(__file__))
from preprocessing import load_raw_ehg_record, apply_bandpass_filter


def extract_imf_features_window(win_60s: np.ndarray, fs: float = 20.0) -> dict:
    """
    Decompose 60-second window across 3 channels into first 4 IMFs
    and extract 16 features per channel = 48 features total.
    """
    emd = EMD()
    feats = {}
    for ch_idx in range(3):
        ch_sig = win_60s[:, ch_idx]
        try:
            imfs = emd(ch_sig, max_imf=4)
        except Exception:
            imfs = np.array([ch_sig])
            
        n_imfs = imfs.shape[0]
        total_energy = np.sum([np.mean(imfs[i]**2) for i in range(n_imfs)]) + 1e-10
        
        for imf_idx in range(4):
            prefix = f'ch{ch_idx+1}_imf{imf_idx+1}'
            if imf_idx < n_imfs:
                imf = imfs[imf_idx]
                eng = float(np.mean(imf**2))
                rel_eng = float(eng / total_energy)
                
                # Median frequency
                freqs, psd = signal.welch(imf, fs=fs, nperseg=min(256, len(imf)))
                cum_power = np.cumsum(psd)
                med_idx = np.where(cum_power >= cum_power[-1] * 0.5)[0]
                f_med = float(freqs[med_idx[0]]) if len(med_idx) > 0 else 0.0
                
                # Sample entropy
                try:
                    se = float(ant.sample_entropy(imf))
                    if np.isnan(se) or np.isinf(se):
                        se = 0.0
                except Exception:
                    se = 0.0
            else:
                eng, rel_eng, f_med, se = 0.0, 0.0, 0.0, 0.0
                
            feats[f'{prefix}_energy'] = eng
            feats[f'{prefix}_rel_energy'] = rel_eng
            feats[f'{prefix}_f_med'] = f_med
            feats[f'{prefix}_sampen'] = se
            
    return feats


def extract_and_cache_imf_dataset(
    burst_csv: str,
    raw_root: str,
    output_csv: str,
    dataset_name: str = 'TPEHGDB'
) -> pd.DataFrame:
    """Extract IMF features for all records in the burst manifest and save to CSV."""
    if os.path.exists(output_csv):
        print(f"Loading cached IMF features from {output_csv}...")
        return pd.read_csv(output_csv)

    print(f"Extracting IMF features for {dataset_name} from {raw_root}...")
    df_burst = pd.read_csv(burst_csv)
    records = []
    
    t0 = time.time()
    for idx, r in df_burst.iterrows():
        rec_id = r['record_id']
        rec_path = os.path.join(raw_root, rec_id)
        sig, fs, _ = load_raw_ehg_record(rec_path, dataset_name)
        filt = apply_bandpass_filter(sig, 0.3, 3.0, fs)
        
        start_idx = int(r['start_time_sec'] * fs)
        win = filt[start_idx:start_idx + int(60 * fs), :]
        
        imf_dict = extract_imf_features_window(win, fs=fs)
        imf_dict['record_id'] = rec_id
        imf_dict['pregnancy_id'] = r.get('pregnancy_id', rec_id)
        imf_dict['label'] = r['label']
        records.append(imf_dict)
        
        if (idx + 1) % 50 == 0 or (idx + 1) == len(df_burst):
            print(f"Processed {idx + 1}/{len(df_burst)} records ({(time.time()-t0):.1f}s)...")
            
    df_out = pd.DataFrame(records)
    df_out.to_csv(output_csv, index=False)
    print(f"Saved {len(df_out)} records to {output_csv}")
    return df_out


def run_cv_evaluation(X, y, cv, models_dict):
    """Run 5-fold CV evaluation across a dictionary of models."""
    results = []
    for name, clf in models_dict.items():
        preds = np.zeros(len(y))
        probs = np.zeros(len(y))
        
        for train_idx, val_idx in cv.split(X, y):
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]
            
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_va_sc = scaler.transform(X_va)
            
            if name.startswith('Top-20 ANOVA'):
                sel = SelectKBest(f_classif, k=min(20, X.shape[1]))
                X_tr_sub = sel.fit_transform(X_tr_sc, y_tr)
                X_va_sub = sel.transform(X_va_sc)
                m = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42)
                m.fit(X_tr_sub, y_tr)
                preds[val_idx] = m.predict(X_va_sub)
                probs[val_idx] = m.predict_proba(X_va_sub)[:, 1]
            else:
                clf.fit(X_tr_sc, y_tr)
                preds[val_idx] = clf.predict(X_va_sc)
                if hasattr(clf, 'predict_proba'):
                    probs[val_idx] = clf.predict_proba(X_va_sc)[:, 1]
                else:
                    probs[val_idx] = clf.decision_function(X_va_sc)
                    
        acc = accuracy_score(y, preds)
        bal_acc = balanced_accuracy_score(y, preds)
        sens = recall_score(y, preds, pos_label=1)
        spec = recall_score(y, preds, pos_label=0)
        try:
            auc = roc_auc_score(y, probs)
        except Exception:
            auc = 0.5
            
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Balanced_Acc': bal_acc,
            'Sensitivity': sens,
            'Specificity': spec,
            'ROC_AUC': auc
        })
    return pd.DataFrame(results)


def main():
    base_dir = r'c:\Users\piyus\Downloads\ML 2\term-preterm-ehg-database-1.0.1\ehg_triage'
    tpehgdb_root = r'c:\Users\piyus\Downloads\ML 2\term-preterm-ehg-database-1.0.1\term-preterm-ehg-database-1.0.1\tpehgdb'
    tpehgt_root = r'c:\Users\piyus\Downloads\ML 2\term-preterm-ehg-dataset-with-tocogram-1.0.0\term-preterm-ehg-dataset-with-tocogram-1.0.0'
    data_dir = os.path.join(base_dir, 'data')
    reports_dir = os.path.join(base_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    burst_tpehgdb = os.path.join(data_dir, 'features_60s_burst_detected_all.csv')
    burst_tpehgt = os.path.join(data_dir, 'features_60s_burst_detected_tpehgt13.csv')
    
    out_imf_tpehgdb = os.path.join(data_dir, 'features_imf_48_tpehgdb.csv')
    out_imf_tpehgt = os.path.join(data_dir, 'features_imf_48_tpehgt13.csv')
    
    # 1. Extract / Cache IMF Features
    df_imf_db = extract_and_cache_imf_dataset(burst_tpehgdb, tpehgdb_root, out_imf_tpehgdb, 'TPEHGDB')
    df_imf_gt = extract_and_cache_imf_dataset(burst_tpehgt, tpehgt_root, out_imf_tpehgt, 'TPEHGT')
    
    # 2. Load Original 69 Self-Normalized Features
    df_orig = pd.read_csv(os.path.join(data_dir, 'features_69_self_normalized.csv'))
    
    # Define feature matrices
    meta_cols = ['record_id', 'pregnancy_id', 'label']
    imf_cols = [c for c in df_imf_db.columns if c not in meta_cols]
    orig_cols = [c for c in df_orig.columns if c not in meta_cols]
    
    # Merged dataset: Original 69 + IMF 48 = 117 features
    df_merged = pd.merge(df_orig, df_imf_db[['record_id'] + imf_cols], on='record_id')
    merged_cols = orig_cols + imf_cols
    
    y = df_orig['label'].values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Define models to benchmark
    models = {
        'ExtraTrees (depth=3, balanced)': ExtraTreesClassifier(n_estimators=100, max_depth=3, min_samples_leaf=5, class_weight='balanced', random_state=42),
        'Logistic Regression (C=0.5, balanced)': LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42),
        'Gaussian RBF SVM (gamma=0.001, balanced)': SVC(C=1.0, kernel='rbf', gamma=0.001, class_weight='balanced', probability=True, random_state=42),
        'Top-20 ANOVA + Logistic Regression': None
    }
    
    print("\n--- BENCHMARK 1: Standalone IMF Features (48 feats) ---")
    X_imf = df_imf_db[imf_cols].values
    res_imf = run_cv_evaluation(X_imf, y, cv, models)
    res_imf['Feature_Set'] = 'Standalone IMF (48)'
    print(res_imf.to_string(index=False))
    
    print("\n--- BENCHMARK 2: Current Baseline Features (69 feats) ---")
    X_orig = df_orig[orig_cols].values
    res_orig = run_cv_evaluation(X_orig, y, cv, models)
    res_orig['Feature_Set'] = 'Current Self-Normalized (69)'
    print(res_orig.to_string(index=False))
    
    print("\n--- BENCHMARK 3: Hybrid Fusion (Original 69 + IMF 48 = 117 feats) ---")
    X_merged = df_merged[merged_cols].values
    res_merged = run_cv_evaluation(X_merged, y, cv, models)
    res_merged['Feature_Set'] = 'Hybrid Fusion (117)'
    print(res_merged.to_string(index=False))
    
    all_res = pd.concat([res_imf, res_orig, res_merged], ignore_index=True)
    report_csv = os.path.join(reports_dir, 'emd_imf_evaluation_report.csv')
    all_res.to_csv(report_csv, index=False)
    print(f"\nSaved full benchmark comparison to {report_csv}")


if __name__ == '__main__':
    main()
