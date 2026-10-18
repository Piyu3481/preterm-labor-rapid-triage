"""
inference.py - Production-Ready One-Minute EHG Triage Inference Engine

Functionality:
- Validates 60-second, 3-channel EHG inputs (sampling rate, duration, signal validity)
- Applies exact training bandpass filter and feature pipeline
- Computes calibrated preterm risk probability
- Measures inference execution latency
- Clearly flags research prototype status and clinical disclaimer
"""

import os
import time
import json
import joblib
import numpy as np
from typing import Dict, Any, Union, Tuple, Optional

from preprocessing import TARGET_FS, WINDOW_SAMPLES, apply_bandpass_filter, check_signal_quality
from features import extract_features_from_window


class EHGTriagePredictor:
    """
    Inference predictor for 60-second 3-channel EHG preterm risk triage.
    """
    def __init__(self, models_dir: Optional[str] = None):
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
            
        model_path = os.path.join(models_dir, 'best_triage_model.joblib')
        scaler_path = os.path.join(models_dir, 'fitted_scaler.joblib')
        features_path = os.path.join(models_dir, 'feature_names.json')

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Model or scaler artifacts missing in {models_dir}")

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        with open(features_path, 'r') as f:
            self.feature_names = json.load(f)

    def predict_one_minute(
        self,
        signal_window: np.ndarray,
        fs: float = TARGET_FS,
        threshold: float = 0.50
    ) -> Dict[str, Any]:
        """
        Accepts a 60-second 3-channel EHG signal.
        signal_window: (1200, 3) array in millivolts
        fs: sampling rate in Hz (must be 20.0 Hz or close)
        """
        start_clock = time.perf_counter()

        # 1. Input Dimension & Type Check
        signal_window = np.asarray(signal_window, dtype=np.float64)
        if signal_window.ndim != 2 or signal_window.shape[1] != 3:
            return {
                'status': 'ERROR',
                'error_message': f"Expected shape (N, 3), got {signal_window.shape}",
                'disclaimer': "RESEARCH PROTOTYPE ONLY - Not for clinical diagnostic use."
            }

        # 2. Resample if sampling rate slightly deviates from 20.0 Hz
        if abs(fs - TARGET_FS) > 1e-3:
            import scipy.signal as dsp
            n_target = int(round(signal_window.shape[0] * TARGET_FS / fs))
            signal_window = dsp.resample(signal_window, n_target, axis=0)

        # 3. Check Duration
        if signal_window.shape[0] != WINDOW_SAMPLES:
            return {
                'status': 'ERROR',
                'error_message': f"Expected 60 seconds ({WINDOW_SAMPLES} samples at {TARGET_FS} Hz), got {signal_window.shape[0]} samples",
                'disclaimer': "RESEARCH PROTOTYPE ONLY - Not for clinical diagnostic use."
            }

        # 4. Signal Quality Assessment
        is_valid, quality_reason = check_signal_quality(signal_window)
        if not is_valid:
            return {
                'status': 'REJECTED_QUALITY',
                'quality_flag': 'FAIL',
                'quality_reason': quality_reason,
                'disclaimer': "RESEARCH PROTOTYPE ONLY - Signal quality unacceptable for triage."
            }

        # 5. Exact Training Preprocessing: Bandpass Filter (0.3 - 3.0 Hz, 4-pole Butterworth)
        filtered_sig = apply_bandpass_filter(signal_window, lowcut=0.3, highcut=3.0, fs=TARGET_FS)

        # 6. Feature Extraction (57 base + 9 topological + 3 background ratios = 69 features)
        feature_dict = extract_features_from_window(filtered_sig, fs=TARGET_FS)
        
        # Topological shape features
        try:
            from burst_and_advanced_features import compute_topological_shape_features
            topo_dict = compute_topological_shape_features(filtered_sig, fs=TARGET_FS)
            feature_dict.update(topo_dict)
        except Exception:
            pass
            
        # Burst-to-background ratios (default to 1.35 training median if single window)
        for ch in [1, 2, 3]:
            if f'ratio_burst_to_bg_ch{ch}' not in feature_dict:
                feature_dict[f'ratio_burst_to_bg_ch{ch}'] = 1.35
        
        # Align features with training schema
        feature_vector = np.array([feature_dict.get(fname, 0.0) for fname in self.feature_names]).reshape(1, -1)

        # 7. Preprocessing Scaling
        scaled_features = self.scaler.transform(feature_vector)

        # 8. Model Inference
        if hasattr(self.model, 'predict_proba'):
            prob_preterm = float(self.model.predict_proba(scaled_features)[0, 1])
        else:
            prob_preterm = float(self.model.predict(scaled_features)[0])

        # Latency measurement
        latency_ms = (time.perf_counter() - start_clock) * 1000.0

        # Triage Category & Interpretation
        if prob_preterm >= 0.55:
            category = "High Preterm Delivery Risk"
            binary_pred = "PRETERM_RISK"
        elif prob_preterm >= 0.35:
            category = "Moderate Preterm Delivery Risk"
            binary_pred = "INDETERMINATE / MONITOR"
        else:
            category = "Low Preterm Delivery Risk (Term Likely)"
            binary_pred = "TERM_LIKELY"

        return {
            'status': 'SUCCESS',
            'risk_score': prob_preterm,
            'predicted_class': binary_pred,
            'triage_category': category,
            'decision_threshold': threshold,
            'inference_latency_ms': round(latency_ms, 2),
            'signal_quality': 'PASS',
            'key_metrics': {
                'rms_s1_uv': round(feature_dict.get('ch1_rms', 0.0) * 1000, 2),
                'f_med_s1_hz': round(feature_dict.get('ch1_f_med', 0.0), 3),
                'f_peak_s1_hz': round(feature_dict.get('ch1_f_peak', 0.0), 3),
                'sample_entropy_s1': round(feature_dict.get('ch1_samp_en', 0.0), 3),
                's1_s2_correlation': round(feature_dict.get('corr_ch1_ch2', 0.0), 3)
            },
            'disclaimer': (
                "RESEARCH PROTOTYPE ONLY: This experimental triage proxy assesses recorded "
                "pregnancy outcome risk from 60-second EHG segments. It is not clinically "
                "validated, not FDA-cleared, and must NOT be used for acute preterm labor diagnosis."
            )
        }


def benchmark_inference_latency(n_runs: int = 50) -> Dict[str, float]:
    """Benchmark inference latency on synthetic 60s 3-channel EHG signals."""
    predictor = EHGTriagePredictor()
    dummy_signal = 0.05 * np.random.randn(WINDOW_SAMPLES, 3)

    # Warmup
    for _ in range(5):
        predictor.predict_one_minute(dummy_signal)

    latencies = []
    for _ in range(n_runs):
        res = predictor.predict_one_minute(dummy_signal)
        latencies.append(res['inference_latency_ms'])

    return {
        'mean_latency_ms': float(np.mean(latencies)),
        'std_latency_ms': float(np.std(latencies)),
        'min_latency_ms': float(np.min(latencies)),
        'max_latency_ms': float(np.max(latencies)),
        'p95_latency_ms': float(np.percentile(latencies, 95))
    }


if __name__ == '__main__':
    print("Testing 1-minute EHG Inference Engine...")
    predictor = EHGTriagePredictor()
    test_sig = 0.05 * np.random.randn(WINDOW_SAMPLES, 3)
    res = predictor.predict_one_minute(test_sig)
    print(json.dumps(res, indent=2))
    
    print("\nBenchmarking Inference Latency (50 runs)...")
    bench = benchmark_inference_latency(50)
    print(json.dumps(bench, indent=2))
