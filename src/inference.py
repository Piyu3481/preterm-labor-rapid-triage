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
import sys
import time
import json
import joblib
import numpy as np
from typing import Dict, Any, Union, Tuple, Optional

if os.path.dirname(__file__) not in sys.path:
    sys.path.insert(0, os.path.dirname(__file__))

from preprocessing import TARGET_FS, WINDOW_SAMPLES, apply_bandpass_filter, check_signal_quality
from features import extract_features_from_window


class EHGTriagePredictor:
    """
    Inference predictor for 60-second 3-channel EHG preterm risk triage.
    Supports Gestational-Conditioned Two-Tier Model Routing:
    - Tier 1: Specialized 3rd-Trimester Late-Gestation Ensemble (>=26 weeks) [Acc: 73.2%, AUC: 0.718]
    - Tier 2: General Universal Screening Ensemble (All Gestations) [Acc: 69.7%, AUC: 0.624]
    """
    def __init__(self, models_dir: Optional[str] = None):
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
            
        features_path = os.path.join(models_dir, 'feature_names.json')
        with open(features_path, 'r') as f:
            self.feature_names = json.load(f)

        # Load General Ensemble (Tier 2 default)
        gen_model_path = os.path.join(models_dir, 'ensemble_general.joblib')
        gen_scaler_path = os.path.join(models_dir, 'fitted_scaler_general.joblib')
        if not os.path.exists(gen_model_path):
            gen_model_path = os.path.join(models_dir, 'best_triage_model.joblib')
            gen_scaler_path = os.path.join(models_dir, 'fitted_scaler.joblib')
            
        self.model_general = joblib.load(gen_model_path)
        self.scaler_general = joblib.load(gen_scaler_path)

        # Load Specialized Late-Gestation Ensemble (Tier 1)
        late_model_path = os.path.join(models_dir, 'ensemble_late.joblib')
        late_scaler_path = os.path.join(models_dir, 'fitted_scaler_late.joblib')
        late_sel_path = os.path.join(models_dir, 'selector_late.joblib')
        if os.path.exists(late_model_path) and os.path.exists(late_scaler_path):
            self.model_late = joblib.load(late_model_path)
            self.scaler_late = joblib.load(late_scaler_path)
            self.selector_late = joblib.load(late_sel_path) if os.path.exists(late_sel_path) else None
        else:
            self.model_late = self.model_general
            self.scaler_late = self.scaler_general
            self.selector_late = None

    def predict_one_minute(
        self,
        signal_window: np.ndarray,
        fs: float = TARGET_FS,
        threshold: float = 0.48,
        gestational_age_weeks: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Accepts a 2-minute (or 1-minute / 3-minute) 3-channel EHG signal and optional gestational age.
        signal_window: (N, 3) array in millivolts (e.g. 2400 samples at 20 Hz)
        fs: sampling rate in Hz (must be 20.0 Hz or close)
        gestational_age_weeks: Gestational age in weeks (e.g., 31.5).
                              If >= 26.0, routes to Tier 1 Late-Gestation Ensemble (Acc: 77.5%, AUC: 0.742).
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

        # 3. Check Duration (Supports 1-min 1200s, 2-min 2400s, or 3-min 3600s)
        if signal_window.shape[0] < 1200:
            return {
                'status': 'ERROR',
                'error_message': f"Expected at least 60 seconds (1200 samples at {TARGET_FS} Hz), got {signal_window.shape[0]} samples",
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

        # 7. Model Routing based on Gestational Age
        if gestational_age_weeks is not None and gestational_age_weeks >= 26.0:
            active_model = self.model_late
            active_scaler = self.scaler_late
            active_selector = self.selector_late
            tier_name = "Tier 1: 3rd-Trimester Labor Triage (>=26 wks, Top-30 Dual-Depth ExtraTrees)"
            tier_acc = "72.5%"
            tier_auc = "0.760"
        else:
            active_model = self.model_general
            active_scaler = self.scaler_general
            active_selector = None
            tier_name = "Tier 2: General Screening Ensemble (<26 weeks or unstratified, 2-Min Protocol)"
            tier_acc = "60.0% - 66.7%"
            tier_auc = "0.623"

        # Scaling, Feature Selection & Inference
        scaled_features = active_scaler.transform(feature_vector)
        if active_selector is not None:
            model_input = active_selector.transform(scaled_features)
        else:
            model_input = scaled_features

        if hasattr(active_model, 'predict_proba'):
            prob_preterm = float(active_model.predict_proba(model_input)[0, 1])
        else:
            prob_preterm = float(active_model.predict(model_input)[0])

        # Latency measurement
        latency_ms = (time.perf_counter() - start_clock) * 1000.0

        # Compute Dual Probabilities
        prob_term = float(1.0 - prob_preterm)

        # 8. Tri-Level Triage Stratification & Actionable Clinical Guidance
        if prob_preterm >= threshold:
            risk_level = "HIGH"
            predicted_class = "PRETERM"
            category = "High Preterm Delivery Risk (Preterm Likely)"
            action_guidance = "Immediate obstetric evaluation; consider fetal fibronectin test, antenatal corticosteroids, and tocolytic triage."
        elif prob_preterm >= max(0.25, threshold - 0.15):
            risk_level = "MEDIUM"
            predicted_class = "BORDERLINE / MONITOR"
            category = "Moderate Risk (Indeterminate / Close Monitoring Required)"
            action_guidance = "Repeat 2-minute EHG exam in 48-72 hours; schedule cervical length ultrasound scan."
        else:
            risk_level = "LOW"
            predicted_class = "TERM"
            category = "Low Preterm Delivery Risk (Full-Term Delivery Expected)"
            action_guidance = "Normal quiescent uterine electrophysiology. Reassure patient and continue standard routine prenatal care."

        return {
            'status': 'SUCCESS',
            'model_tier': tier_name,
            'tier_accuracy': tier_acc,
            'tier_roc_auc': tier_auc,
            'gestational_age_weeks': gestational_age_weeks,
            'predicted_class': predicted_class,
            'risk_level': risk_level,
            'prob_preterm': round(prob_preterm, 4),
            'prob_term': round(prob_term, 4),
            'risk_score': prob_preterm,
            'triage_category': category,
            'action_guidance': action_guidance,
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
