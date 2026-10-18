"""
features.py - Feature Extraction Pipeline for 60-second 3-Channel EHG Signals

Features computed per channel:
1. Time Domain:
   - RMS (Root Mean Square voltage)
   - Variance and Standard Deviation
   - Peak-to-Peak Amplitude
   - Skewness and Kurtosis
   - Zero-Crossing Rate
2. Frequency Domain (Welch Power Spectral Density):
   - Peak frequency (Fpeak)
   - Median frequency (Fmed)
   - Spectral Band Powers (0.1-0.3 Hz, 0.3-1.0 Hz, 1.0-2.0 Hz, 2.0-3.0 Hz, Total Power)
   - Fast-to-slow band power ratio (P_0.3_1.0 / P_0.1_0.3)
   - High-to-total band power ratio (P_0.3_3.0 / P_total)
   - Spectral Entropy
3. Non-Linear Complexity:
   - Sample Entropy (m=2, r=0.2*std)
4. Cross-Channel Coupling:
   - Cross-correlation peak between S1-S2, S2-S3, S1-S3
"""

import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import pandas as pd
from typing import Dict, Any, List, Optional


def compute_sample_entropy(x: np.ndarray, m: int = 2, r: Optional[float] = None) -> float:
    """
    Calculate Sample Entropy (SampEn) for a 1D time series.
    m: embedding dimension (default 2)
    r: tolerance threshold (default 0.2 * standard deviation)
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    std_val = np.std(x)
    if std_val < 1e-8 or n <= m + 1:
        return 0.0
    if r is None:
        r = 0.2 * std_val

    # Fast vectorised template matching
    X_m = np.array([x[i:i + m] for i in range(n - m)])
    diff_m = np.abs(X_m[:, None, :] - X_m[None, :, :]).max(axis=2)
    B = (diff_m <= r).sum() - (n - m)

    X_m1 = np.array([x[i:i + m + 1] for i in range(n - m - 1)])
    diff_m1 = np.abs(X_m1[:, None, :] - X_m1[None, :, :]).max(axis=2)
    A = (diff_m1 <= r).sum() - (n - m - 1)

    if B <= 0 or A <= 0:
        return 0.0
    return float(-np.log(A / B))


def compute_spectral_features(x: np.ndarray, fs: float = 20.0) -> Dict[str, float]:
    """
    Compute frequency-domain features from Welch's Power Spectral Density.
    """
    nperseg = min(256, len(x))
    freqs, psd = signal.welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)

    total_power = np.sum(psd)
    if total_power <= 1e-12:
        return {
            'f_peak': 0.0,
            'f_med': 0.0,
            'p_total': 0.0,
            'p_01_03': 0.0,
            'p_03_10': 0.0,
            'p_10_20': 0.0,
            'p_20_30': 0.0,
            'ratio_fast_slow': 0.0,
            'ratio_core_total': 0.0,
            'spectral_entropy': 0.0
        }

    # Peak frequency
    f_peak = float(freqs[np.argmax(psd)])

    # Median frequency
    cum_power = np.cumsum(psd)
    med_idx = np.searchsorted(cum_power, total_power / 2.0)
    f_med = float(freqs[min(med_idx, len(freqs) - 1)])

    # Band powers
    band_01_03 = (freqs >= 0.1) & (freqs < 0.3)
    band_03_10 = (freqs >= 0.3) & (freqs < 1.0)
    band_10_20 = (freqs >= 1.0) & (freqs < 2.0)
    band_20_30 = (freqs >= 2.0) & (freqs <= 3.0)

    p_01_03 = float(np.sum(psd[band_01_03]))
    p_03_10 = float(np.sum(psd[band_03_10]))
    p_10_20 = float(np.sum(psd[band_10_20]))
    p_20_30 = float(np.sum(psd[band_20_30]))

    ratio_fast_slow = float(p_03_10 / (p_01_03 + 1e-8))
    ratio_core_total = float((p_03_10 + p_10_20) / (total_power + 1e-8))

    # Spectral entropy
    prob = psd / total_power
    prob = prob[prob > 0]
    spectral_entropy = float(-np.sum(prob * np.log2(prob)))

    return {
        'f_peak': f_peak,
        'f_med': f_med,
        'p_total': float(total_power),
        'p_01_03': p_01_03,
        'p_03_10': p_03_10,
        'p_10_20': p_10_20,
        'p_20_30': p_20_30,
        'ratio_fast_slow': ratio_fast_slow,
        'ratio_core_total': ratio_core_total,
        'spectral_entropy': spectral_entropy
    }


def compute_temporal_features(x: np.ndarray) -> Dict[str, float]:
    """
    Compute time-domain statistical features.
    """
    rms = float(np.sqrt(np.mean(x ** 2)))
    var = float(np.var(x))
    std = float(np.std(x))
    peak_to_peak = float(np.max(x) - np.min(x))
    skew = float(stats.skew(x)) if std > 1e-8 else 0.0
    kurt = float(stats.kurtosis(x)) if std > 1e-8 else 0.0

    # Zero crossing rate (centered)
    x_centered = x - np.mean(x)
    zero_crossings = np.sum(np.diff(np.sign(x_centered)) != 0) / (len(x) - 1)

    return {
        'rms': rms,
        'var': var,
        'std': std,
        'peak_to_peak': peak_to_peak,
        'skew': skew,
        'kurtosis': kurt,
        'zcr': float(zero_crossings)
    }


def extract_features_from_window(
    sig_window: np.ndarray,
    fs: float = 20.0,
    channel_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Extract comprehensive feature set from a 60-second 3-channel window (N, 3).
    """
    if channel_names is None:
        channel_names = ['ch1', 'ch2', 'ch3']

    features = {}

    for ch_idx, ch_name in enumerate(channel_names):
        x = sig_window[:, ch_idx]

        # Temporal features
        temp = compute_temporal_features(x)
        for k, v in temp.items():
            features[f'{ch_name}_{k}'] = v

        # Spectral features
        spec = compute_spectral_features(x, fs=fs)
        for k, v in spec.items():
            features[f'{ch_name}_{k}'] = v

        # Sample entropy
        features[f'{ch_name}_samp_en'] = compute_sample_entropy(x, m=2)

    # Cross-channel coupling
    for i in range(3):
        for j in range(i + 1, 3):
            ch_a = sig_window[:, i] - np.mean(sig_window[:, i])
            ch_b = sig_window[:, j] - np.mean(sig_window[:, j])
            std_a = np.std(ch_a)
            std_b = np.std(ch_b)
            if std_a > 1e-8 and std_b > 1e-8:
                corr = np.correlate(ch_a / std_a, ch_b / std_b, mode='full') / len(ch_a)
                max_corr = float(np.max(np.abs(corr)))
            else:
                max_corr = 0.0
            features[f'corr_{channel_names[i]}_{channel_names[j]}'] = max_corr

    return features


def load_tpehgdb_baseline_features(fvl_path: str) -> pd.DataFrame:
    """
    Load precomputed baseline features from TPEHGDB .fvl files for reproducibility benchmark.
    """
    rows = []
    with open(fvl_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    header = [h.strip() for h in lines[0].split('|')]
    for line in lines[2:]:
        parts = [p.strip() for p in line.split('|')]
        if len(parts) == len(header):
            rows.append(parts)
    df = pd.DataFrame(rows, columns=header)
    num_cols = ['Gestation', 'Rec. time', 'RMS', 'Fmed', 'Fpeak', 'Samp. en.']
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['is_preterm'] = (df['Premature'] == 't').astype(int)
    return df
