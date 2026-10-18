"""
splits.py - Patient-Level Stratified Cross-Validation Splitting Strategy

Enforces:
- Strict patient/pregnancy-level grouping (StratifiedGroupKFold)
- All windows from a single pregnancy/patient are kept in the same fold
- Prevents cross-dataset leakage between identical/matched records
- Provides 3 evaluation regimes:
  1. TPEHGDB_ONLY (300 records, 5-fold Stratified Patient CV)
  2. COMBINED_GROUPED (Combined dataset with patient clusters kept intact)
  3. TRANSFER_TEST (Train on TPEHGDB, test on non-overlapping independent TPEHGT cohort)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from typing import List, Tuple, Dict, Any


def get_stratified_patient_folds(
    df_manifest: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate stratified patient-level folds.
    Returns list of (train_indices, val_indices).
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Support both string labels ('Preterm'/'Term') and numeric/binary labels (1/0)
    if df_manifest['label'].dtype == object or isinstance(df_manifest['label'].iloc[0], str):
        y = (df_manifest['label'] == 'Preterm').astype(int).values
    else:
        y = (df_manifest['label'] == 1).astype(int).values
    groups = df_manifest['pregnancy_id'].values
    X = np.arange(len(df_manifest))

    folds = []
    for train_idx, val_idx in sgkf.split(X, y, groups):
        folds.append((train_idx, val_idx))

    return folds


def assign_folds_to_manifest(
    df_manifest: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Assign a 'fold' column (0 to n_splits-1) to the manifest dataframe.
    """
    df_out = df_manifest.copy()
    df_out['fold'] = -1
    
    folds = get_stratified_patient_folds(df_out, n_splits=n_splits, random_state=random_state)
    for fold_num, (_, val_idx) in enumerate(folds):
        df_out.iloc[val_idx, df_out.columns.get_loc('fold')] = fold_num

    return df_out


def create_transfer_split(
    df_manifest: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a strict cross-dataset transfer split:
    Train on: all TPEHGDB records (300 records).
    Test on: Independent, non-overlapping TPEHGT pregnant records only (13 records).
    All overlapping TPEHGT records and non-pregnant controls are excluded from test.
    """
    train_mask = df_manifest['source'] == 'TPEHGDB'
    
    # Independent TPEHGT records (no overlap, no twin pregnancy in TPEHGDB, pregnant)
    independent_tpehgt = [
        'tpehgt_p001', 'tpehgt_p004', 'tpehgt_p005', 'tpehgt_p012', 'tpehgt_p013',
        'tpehgt_t001', 'tpehgt_t002', 'tpehgt_t003', 'tpehgt_t004', 'tpehgt_t005',
        'tpehgt_t006', 'tpehgt_t010', 'tpehgt_t013'
    ]
    test_mask = (df_manifest['source'] == 'TPEHGT') & (df_manifest['record_id'].isin(independent_tpehgt))

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    return train_idx, test_idx
