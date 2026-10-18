"""
preprocessing.py - Standardized EHG Signal Preprocessing and Window Extraction Pipeline

Features:
- Standardizes 3-channel EHG signals (S1, S2, S3) from TPEHGDB and TPEHGT
- Standardizes sampling rate to 20.0 Hz
- Configurable Butterworth band-pass filter (default 0.3 - 3.0 Hz, 4-pole bidirectional)
- Handles 180-second transient exclusion for pre-filtered records
- Predefined 60-second segment selection policies ('center', 'max_energy', 'multi_window')
- Robust signal quality checks (clipping, flatlining, NaN/Inf)
"""

import numpy as np
import scipy.signal as signal
import wfdb
import os
from typing import Tuple, Dict, Any, List, Optional


TARGET_FS = 20.0
WINDOW_SEC = 60.0
WINDOW_SAMPLES = int(WINDOW_SEC * TARGET_FS)  # 1200 samples
TRANSIENT_SEC = 180.0  # 180s filter transient at start/end of pre-filtered channels


def design_bandpass_filter(
    lowcut: float = 0.3,
    highcut: float = 3.0,
    fs: float = TARGET_FS,
    order: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """Design a digital Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # Constrain within valid digital frequency interval
    low = max(0.001, min(low, 0.99))
    high = max(low + 0.01, min(high, 0.999))
    b, a = signal.butter(order, [low, high], btype='bandpass')
    return b, a


def apply_bandpass_filter(
    sig: np.ndarray,
    lowcut: float = 0.3,
    highcut: float = 3.0,
    fs: float = TARGET_FS,
    order: int = 4
) -> np.ndarray:
    """
    Apply zero-phase forward-backward Butterworth bandpass filter.
    sig: (N, C) or (N,) array
    """
    b, a = design_bandpass_filter(lowcut, highcut, fs, order)
    if sig.ndim == 1:
        return signal.filtfilt(b, a, sig)
    filtered = np.zeros_like(sig)
    for ch in range(sig.shape[1]):
        filtered[:, ch] = signal.filtfilt(b, a, sig[:, ch])
    return filtered


def check_signal_quality(
    sig_window: np.ndarray,
    max_abs_mv: float = 5.0,
    min_std_mv: float = 0.001
) -> Tuple[bool, str]:
    """
    Verify signal quality for a 60-second 3-channel window.
    Returns (is_valid, reason).
    """
    if sig_window.shape[0] != WINDOW_SAMPLES:
        return False, f"Expected {WINDOW_SAMPLES} samples, got {sig_window.shape[0]}"
    if sig_window.shape[1] != 3:
        return False, f"Expected 3 channels, got {sig_window.shape[1]}"
    if np.isnan(sig_window).any():
        return False, "Signal contains NaN values"
    if np.isinf(sig_window).any():
        return False, "Signal contains Inf values"
    
    # Check for flatlining
    stds = np.std(sig_window, axis=0)
    if (stds < min_std_mv).any():
        return False, f"Channel flatline detected: stds = {stds}"
    
    # Check for extreme clipping or sensor detachment
    max_val = np.max(np.abs(sig_window))
    if max_val > max_abs_mv:
        return False, f"Sensor saturation detected: max abs = {max_val:.2f} mV"
    
    return True, "Valid"


def load_raw_ehg_record(
    record_path: str,
    source: str
) -> Tuple[np.ndarray, float, List[str]]:
    """
    Load raw 3-channel EHG signal (S1, S2, S3) from WFDB record in physical units (mV).
    TPEHGDB: channels 0, 4, 8 are raw S1, S2, S3
    TPEHGT: channels 0, 2, 4 are raw S1, S2, S3
    """
    if source.upper() == 'TPEHGDB':
        channels = [0, 4, 8]
        sig_names = ['S1', 'S2', 'S3']
    elif source.upper() == 'TPEHGT':
        channels = [0, 2, 4]
        sig_names = ['EHG1', 'EHG2', 'EHG3']
    else:
        raise ValueError(f"Unknown source: {source}")

    sig, fields = wfdb.rdsamp(record_path, channels=channels)
    fs = float(fields['fs'])
    
    # Resample to exactly 20.0 Hz if slight floating mismatch
    if abs(fs - TARGET_FS) > 1e-3:
        num_target = int(round(sig.shape[0] * TARGET_FS / fs))
        sig = signal.resample(sig, num_target, axis=0)
        fs = TARGET_FS

    return sig, fs, sig_names


def extract_60s_segments(
    sig: np.ndarray,
    fs: float = TARGET_FS,
    policy: str = 'center',
    lowcut: float = 0.3,
    highcut: float = 3.0,
    filter_sig: bool = True
) -> List[Dict[str, Any]]:
    """
    Extract valid 60-second segments from a full recording according to a predefined policy.
    
    Policies:
    - 'center': Single 60s window centered in the middle of the recording
    - 'max_energy': 60s window within [180s, T-180s] having highest RMS energy
    - 'multi_window': Multiple non-overlapping 60s windows within [180s, T-180s]
    """
    n_samples = sig.shape[0]
    total_sec = n_samples / fs
    
    if filter_sig:
        filtered_sig = apply_bandpass_filter(sig, lowcut=lowcut, highcut=highcut, fs=fs)
    else:
        filtered_sig = sig

    transient_samples = int(TRANSIENT_SEC * fs)
    valid_start_idx = transient_samples
    valid_end_idx = n_samples - transient_samples - WINDOW_SAMPLES

    if valid_end_idx <= valid_start_idx:
        # If signal is too short to exclude 180s transients, use whatever margin is available
        valid_start_idx = int(0.1 * n_samples)
        valid_end_idx = n_samples - WINDOW_SAMPLES - int(0.1 * n_samples)

    segments = []

    if policy == 'center':
        center_idx = n_samples // 2
        start_idx = center_idx - (WINDOW_SAMPLES // 2)
        end_idx = start_idx + WINDOW_SAMPLES
        window = filtered_sig[start_idx:end_idx, :]
        is_valid, reason = check_signal_quality(window)
        segments.append({
            'start_sample': start_idx,
            'start_time_sec': start_idx / fs,
            'policy': 'center',
            'signal': window,
            'is_valid': is_valid,
            'quality_reason': reason
        })

    elif policy == 'max_energy':
        # Search for the 60s window with maximum RMS in the valid region (stride 10s = 200 samples)
        stride = int(10 * fs)
        best_rms = -1.0
        best_window = None
        best_start = valid_start_idx
        best_valid = False
        best_reason = ""

        for s_idx in range(valid_start_idx, valid_end_idx, stride):
            e_idx = s_idx + WINDOW_SAMPLES
            cand = filtered_sig[s_idx:e_idx, :]
            is_valid, reason = check_signal_quality(cand)
            if is_valid:
                cand_rms = np.sqrt(np.mean(cand ** 2))
                if cand_rms > best_rms:
                    best_rms = cand_rms
                    best_window = cand
                    best_start = s_idx
                    best_valid = True
                    best_reason = reason

        if best_window is None:
            # Fallback to center window
            center_idx = n_samples // 2
            start_idx = center_idx - (WINDOW_SAMPLES // 2)
            best_window = filtered_sig[start_idx:start_idx+WINDOW_SAMPLES, :]
            best_start = start_idx
            best_valid, best_reason = check_signal_quality(best_window)

        segments.append({
            'start_sample': best_start,
            'start_time_sec': best_start / fs,
            'policy': 'max_energy',
            'signal': best_window,
            'is_valid': best_valid,
            'quality_reason': best_reason
        })

    elif policy == 'multi_window':
        # Extract windows spaced by 120s across the valid region
        stride = int(120 * fs)
        for s_idx in range(valid_start_idx, valid_end_idx, stride):
            e_idx = s_idx + WINDOW_SAMPLES
            w = filtered_sig[s_idx:e_idx, :]
            is_valid, reason = check_signal_quality(w)
            segments.append({
                'start_sample': s_idx,
                'start_time_sec': s_idx / fs,
                'policy': 'multi_window',
                'signal': w,
                'is_valid': is_valid,
                'quality_reason': reason
            })

    else:
        raise ValueError(f"Unknown segment selection policy: {policy}")

    return segments
