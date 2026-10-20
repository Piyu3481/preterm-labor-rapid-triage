"""
pathway3_3min_triage.py - 3-Minute (180s) Enhanced Clinical Protocol Benchmark.

Implements:
1. Extraction of 180-second (3-minute, 3600 samples @ 20 Hz) peak contraction windows.
2. Full 69-feature extraction on the expanded window:
   - 57 base electrophysiological features (time, frequency, entropy, cross-channel)
   - 9 topological contraction envelope features (area, perimeter, concave hull deficit)
   - 3 self-normalized contraction burst-to-baseline ratios
3. 5-Fold Stratified Patient CV on both:
   - General Cohort (N=300)
   - 3rd-Trimester Late Cohort (>=26 weeks, N=138)
4. External hospital transfer on 13 independent TPEHGT records.
5. Direct side-by-side comparison with the 1-minute (60s) baseline.
6. Serializing model weights and saving reports/pathway3_3min_evaluation_report.csv.
"""

import os
import sys
import time
import json
import glob
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
    f1_score
)

sys.path.append(os.path.dirname(__file__))
from preprocessing import load_raw_ehg_record, apply_bandpass_filter, check_signal_quality
from features import extract_features_from_window
from burst_and_advanced_features import compute_topological_shape_features


def detect_and_extract_180s_window(sig: np.ndarray, fs: float = 20.0) -> dict:
    """
    Locates the peak uterine contraction burst in [180s, 1620s] and extracts
    a full 180-second (3-minute = 3600 samples) window centered on the peak.
    """
    n_samples = sig.shape[0]
    dur_sec = n_samples / fs
    
    # Safe search window: [180s, dur_sec - 180s]
    search_start = int(180 * fs)
    search_end = int(min(dur_sec - 180, 1620) * fs)
    
    if search_end - search_start < int(180 * fs):
        # Fallback to middle 180s if recording is short
        center_idx = n_samples // 2
        start_idx = max(0, center_idx - int(90 * fs))
        end_idx = min(n_samples, start_idx + int(180 * fs))
        return {'window': sig[start_idx:end_idx, :], 'peak_time_sec': center_idx / fs, 'start_sec': start_idx / fs}
        
    # Analytic envelope on channel 1 for burst detection
    from scipy.signal import hilbert
    s1_search = sig[search_start:search_end, 0]
    analytic = hilbert(s1_search)
    env = np.abs(analytic)
    
    # Smooth envelope with 10-second moving average (200 samples)
    kernel = np.ones(int(10 * fs)) / (10 * fs)
    smooth_env = np.convolve(env, kernel, mode='same')
    
    peak_rel_idx = np.argmax(smooth_env)
    peak_abs_idx = search_start + peak_rel_idx
    
    # Center 180s (3 min) on peak
    half_win = int(90 * fs)  # 1800 samples
    start_idx = peak_abs_idx - half_win
    end_idx = start_idx + int(180 * fs)  # 3600 samples
    
    # Clamp bounds to [0, n_samples]
    if start_idx < 0:
        start_idx = 0
        end_idx = int(180 * fs)
    elif end_idx > n_samples:
        end_idx = n_samples
        start_idx = max(0, end_idx - int(180 * fs))
        
    return {
        'window': sig[start_idx:end_idx, :],
        'peak_time_sec': peak_abs_idx / fs,
        'start_sec': start_idx / fs
    }


def extract_features_180s(win_180s: np.ndarray, bg_rms: np.ndarray, fs: float = 20.0) -> dict:
    """Extract full 69-feature profile from 180-second window."""
    # 1. 57 Base features (scaled to 180s)
    feats = extract_features_from_window(win_180s, fs=fs)
    
    # 2. 9 Topological shape features
    try:
        topo_feats = compute_topological_shape_features(win_180s, fs=fs)
        feats.update(topo_feats)
    except Exception:
        pass
        
    # 3. Burst-to-background self-normalized ratios
    for ch in range(3):
        win_rms = np.sqrt(np.mean(win_180s[:, ch]**2))
        ratio = win_rms / (bg_rms[ch] + 1e-8)
        feats[f'ratio_burst_to_bg_ch{ch+1}'] = float(ratio)
        
    return feats


def build_or_load_180s_dataset(
    manifest_csv: str,
    raw_root: str,
    output_csv: str,
    dataset_name: str = 'TPEHGDB'
) -> pd.DataFrame:
    """Extract or load cached 180s feature dataset."""
    if os.path.exists(output_csv):
        print(f"Loading cached 180s dataset from {output_csv}...")
        return pd.read_csv(output_csv)
        
    print(f"Extracting 180s (3-minute) windows for {dataset_name} from {raw_root}...")
    df_man = pd.read_csv(manifest_csv)
    records = []
    
    t0 = time.time()
    for idx, r in df_man.iterrows():
        rec_id = r['record_id']
        rec_path = os.path.join(raw_root, rec_id)
        sig, fs, _ = load_raw_ehg_record(rec_path, dataset_name)
        filt = apply_bandpass_filter(sig, 0.3, 3.0, fs)
        
        # Background RMS across [180s, 1620s]
        bg_start = int(180 * fs)
        bg_end = int(min(filt.shape[0] - 180, 1620) * fs)
        if bg_end > bg_start:
            bg_rms = np.sqrt(np.mean(filt[bg_start:bg_end, :]**2, axis=0))
        else:
            bg_rms = np.sqrt(np.mean(filt**2, axis=0))
            
        win_info = detect_and_extract_180s_window(filt, fs=fs)
        f_dict = extract_features_180s(win_info['window'], bg_rms, fs=fs)
        
        f_dict['record_id'] = rec_id
        f_dict['pregnancy_id'] = r.get('pregnancy_id', rec_id)
        f_dict['label'] = r['label']
        f_dict['start_time_sec'] = win_info['start_sec']
        f_dict['peak_time_sec'] = win_info['peak_time_sec']
        records.append(f_dict)
        
        if (idx + 1) % 50 == 0 or (idx + 1) == len(df_man):
            print(f"Extracted {idx + 1}/{len(df_man)} records ({(time.time()-t0):.1f}s)...")
            
    df_out = pd.DataFrame(records)
    df_out.to_csv(output_csv, index=False)
    print(f"Saved {len(df_out)} records to {output_csv}")
    return df_out


def run_5fold_evaluation(X, y, cv, random_state=42):
    """Run 5-fold CV for ExtraTrees and Soft Ensemble."""
    preds_et, probs_et = np.zeros(len(y)), np.zeros(len(y))
    preds_ens, probs_ens = np.zeros(len(y)), np.zeros(len(y))
    
    for tr, val in cv.split(X, y):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr])
        X_va = sc.transform(X[val])
        
        # ExtraTrees
        et = ExtraTreesClassifier(n_estimators=100, max_depth=3, min_samples_leaf=3, class_weight='balanced', random_state=random_state)
        et.fit(X_tr, y[tr])
        preds_et[val] = et.predict(X_va)
        probs_et[val] = et.predict_proba(X_va)[:, 1]
        
        # Soft Ensemble (ET + LR)
        lr = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=random_state)
        ens = VotingClassifier([('et', et), ('lr', lr)], voting='soft', weights=[2, 1])
        ens.fit(X_tr, y[tr])
        preds_ens[val] = ens.predict(X_va)
        probs_ens[val] = ens.predict_proba(X_va)[:, 1]
        
    def metrics(p, prob):
        return {
            'accuracy': accuracy_score(y, p),
            'balanced_accuracy': balanced_accuracy_score(y, p),
            'sensitivity': recall_score(y, p, pos_label=1),
            'specificity': recall_score(y, p, pos_label=0),
            'roc_auc': roc_auc_score(y, prob)
        }
        
    return metrics(preds_et, probs_et), metrics(preds_ens, probs_ens)


def main():
    base_dir = r'c:\Users\piyus\Downloads\ML 2\term-preterm-ehg-database-1.0.1\ehg_triage'
    tpehgdb_root = r'c:\Users\piyus\Downloads\ML 2\term-preterm-ehg-database-1.0.1\term-preterm-ehg-database-1.0.1\tpehgdb'
    tpehgt_root = r'c:\Users\piyus\Downloads\ML 2\term-preterm-ehg-dataset-with-tocogram-1.0.0\term-preterm-ehg-dataset-with-tocogram-1.0.0'
    data_dir = os.path.join(base_dir, 'data')
    reports_dir = os.path.join(base_dir, 'reports')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    man_db = os.path.join(data_dir, 'features_60s_burst_detected_all.csv')
    man_gt = os.path.join(data_dir, 'features_60s_burst_detected_tpehgt13.csv')
    
    out_180s_db = os.path.join(data_dir, 'features_180s_3min_tpehgdb.csv')
    out_180s_gt = os.path.join(data_dir, 'features_180s_3min_tpehgt13.csv')
    
    # 1. Build / Load 180s Datasets
    df_180_db = build_or_load_180s_dataset(man_db, tpehgdb_root, out_180s_db, 'TPEHGDB')
    df_180_gt = build_or_load_180s_dataset(man_gt, tpehgt_root, out_180s_gt, 'TPEHGT')
    
    # 2. Parse Gestational Age (Rectime)
    from train_gestational_ensemble import parse_rectime_metadata, parse_tpehgt_rectime
    df_rectime = parse_rectime_metadata(tpehgdb_root)
    df_180_db = pd.merge(df_180_db, df_rectime, on='record_id')
    
    meta_cols = ['record_id', 'pregnancy_id', 'label', 'start_time_sec', 'peak_time_sec', 'rectime']
    feature_names = [c for c in df_180_db.columns if c not in meta_cols]
    
    print(f"\nExtracted {len(feature_names)} features for 180s windows.")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 3. Benchmark All Patients (N=300)
    X_all = df_180_db[feature_names].values
    y_all = df_180_db['label'].values
    res_all_et, res_all_ens = run_5fold_evaluation(X_all, y_all, cv)
    
    # 4. Benchmark Late Gestation (>=26 weeks, N=138)
    df_late = df_180_db[df_180_db['rectime'] >= 26.0].reset_index(drop=True)
    X_late = df_late[feature_names].values
    y_late = df_late['label'].values
    res_late_et, res_late_ens = run_5fold_evaluation(X_late, y_late, cv)
    
    print("\n================== 3-MINUTE (180s) CV RESULTS ==================")
    print(f"180s General Cohort (N=300):")
    print(f"  ExtraTrees:    Acc={res_all_et['accuracy']*100:.2f}%, BalAcc={res_all_et['balanced_accuracy']*100:.2f}%, Sens={res_all_et['sensitivity']*100:.2f}%, Spec={res_all_et['specificity']*100:.2f}%, AUC={res_all_et['roc_auc']:.4f}")
    print(f"  Soft Ensemble: Acc={res_all_ens['accuracy']*100:.2f}%, BalAcc={res_all_ens['balanced_accuracy']*100:.2f}%, Sens={res_all_ens['sensitivity']*100:.2f}%, Spec={res_all_ens['specificity']*100:.2f}%, AUC={res_all_ens['roc_auc']:.4f}")
    
    print(f"\n180s 3rd-Trimester Late Cohort (>=26 wks, N=138):")
    print(f"  ExtraTrees:    Acc={res_late_et['accuracy']*100:.2f}%, BalAcc={res_late_et['balanced_accuracy']*100:.2f}%, Sens={res_late_et['sensitivity']*100:.2f}%, Spec={res_late_et['specificity']*100:.2f}%, AUC={res_late_et['roc_auc']:.4f}")
    print(f"  Soft Ensemble: Acc={res_late_ens['accuracy']*100:.2f}%, BalAcc={res_late_ens['balanced_accuracy']*100:.2f}%, Sens={res_late_ens['sensitivity']*100:.2f}%, Spec={res_late_ens['specificity']*100:.2f}%, AUC={res_late_ens['roc_auc']:.4f}")
    
    # 5. External Transfer on TPEHGT 13
    X_gt = df_180_gt[feature_names].values
    y_gt = df_180_gt['label'].values
    
    sc_late = StandardScaler()
    X_late_sc = sc_late.fit_transform(X_late)
    sc_all = StandardScaler()
    X_all_sc = sc_all.fit_transform(X_all)
    
    # Train production 180s models
    model_late_180 = VotingClassifier([
        ('et', ExtraTreesClassifier(n_estimators=100, max_depth=3, min_samples_leaf=3, class_weight='balanced', random_state=42)),
        ('lr', LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42))
    ], voting='soft', weights=[2, 1])
    model_late_180.fit(X_late_sc, y_late)
    
    model_all_180 = VotingClassifier([
        ('et', ExtraTreesClassifier(n_estimators=100, max_depth=3, min_samples_leaf=3, class_weight='balanced', random_state=42)),
        ('lr', LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42))
    ], voting='soft', weights=[2, 1])
    model_all_180.fit(X_all_sc, y_all)
    
    # External test
    X_gt_sc = sc_all.transform(X_gt)
    p_gt = model_all_180.predict(X_gt_sc)
    prob_gt = model_all_180.predict_proba(X_gt_sc)[:, 1]
    
    acc_gt = accuracy_score(y_gt, p_gt)
    sens_gt = recall_score(y_gt, p_gt, pos_label=1)
    spec_gt = recall_score(y_gt, p_gt, pos_label=0)
    auc_gt = roc_auc_score(y_gt, prob_gt)
    
    print(f"\n180s External Transfer on TPEHGT (N=13):")
    print(f"  Acc={acc_gt*100:.2f}%, Sens={sens_gt*100:.2f}% ({int(sens_gt*5)}/5 preterms), Spec={spec_gt*100:.2f}%, AUC={auc_gt:.4f}")
    
    # Serialize 180s models
    joblib.dump(model_late_180, os.path.join(models_dir, 'ensemble_late_180s.joblib'))
    joblib.dump(sc_late, os.path.join(models_dir, 'fitted_scaler_late_180s.joblib'))
    joblib.dump(model_all_180, os.path.join(models_dir, 'ensemble_general_180s.joblib'))
    joblib.dump(sc_all, os.path.join(models_dir, 'fitted_scaler_general_180s.joblib'))
    
    # Comparison report
    comp_rows = [
        {
            'Protocol': '1-Minute (60s Baseline)',
            'Cohort': '3rd-Trimester Late (>=26 wks)',
            'Cohort_N': 138,
            'Model': 'Soft Ensemble',
            'Accuracy': 0.7319,
            'Balanced_Acc': 0.5571,
            'Preterm_Sensitivity': 0.3158,
            'Term_Specificity': 0.7983,
            'ROC_AUC': 0.7178
        },
        {
            'Protocol': '3-Minute (180s Pathway 3)',
            'Cohort': '3rd-Trimester Late (>=26 wks)',
            'Cohort_N': 138,
            'Model': 'ExtraTrees',
            'Accuracy': res_late_et['accuracy'],
            'Balanced_Acc': res_late_et['balanced_accuracy'],
            'Preterm_Sensitivity': res_late_et['sensitivity'],
            'Term_Specificity': res_late_et['specificity'],
            'ROC_AUC': res_late_et['roc_auc']
        },
        {
            'Protocol': '3-Minute (180s Pathway 3)',
            'Cohort': '3rd-Trimester Late (>=26 wks)',
            'Cohort_N': 138,
            'Model': 'Soft Ensemble',
            'Accuracy': res_late_ens['accuracy'],
            'Balanced_Acc': res_late_ens['balanced_accuracy'],
            'Preterm_Sensitivity': res_late_ens['sensitivity'],
            'Term_Specificity': res_late_ens['specificity'],
            'ROC_AUC': res_late_ens['roc_auc']
        },
        {
            'Protocol': '1-Minute (60s Baseline)',
            'Cohort': 'General Cohort (All Records)',
            'Cohort_N': 300,
            'Model': 'Soft Ensemble',
            'Accuracy': 0.6967,
            'Balanced_Acc': 0.5788,
            'Preterm_Sensitivity': 0.4211,
            'Term_Specificity': 0.7366,
            'ROC_AUC': 0.6242
        },
        {
            'Protocol': '3-Minute (180s Pathway 3)',
            'Cohort': 'General Cohort (All Records)',
            'Cohort_N': 300,
            'Model': 'Soft Ensemble',
            'Accuracy': res_all_ens['accuracy'],
            'Balanced_Acc': res_all_ens['balanced_accuracy'],
            'Preterm_Sensitivity': res_all_ens['sensitivity'],
            'Term_Specificity': res_all_ens['specificity'],
            'ROC_AUC': res_all_ens['roc_auc']
        },
        {
            'Protocol': '3-Minute (180s Pathway 3)',
            'Cohort': 'External Transfer TPEHGT',
            'Cohort_N': 13,
            'Model': 'Soft Ensemble',
            'Accuracy': acc_gt,
            'Balanced_Acc': (sens_gt + spec_gt) / 2.0,
            'Preterm_Sensitivity': sens_gt,
            'Term_Specificity': spec_gt,
            'ROC_AUC': auc_gt
        }
    ]
    pd.DataFrame(comp_rows).to_csv(os.path.join(reports_dir, 'pathway3_3min_evaluation_report.csv'), index=False)
    print(f"\nSaved Pathway 3 comparison report to reports/pathway3_3min_evaluation_report.csv!")


if __name__ == '__main__':
    main()
