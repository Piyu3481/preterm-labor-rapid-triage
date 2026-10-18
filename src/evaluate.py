"""
evaluate.py - Metrics Computation, Uncertainty Estimation, and Visualization

Calculates:
- Accuracy, Balanced Accuracy
- Sensitivity / Recall (Preterm, Class 1)
- Specificity (Term, Class 0)
- Precision, F1-Score
- ROC-AUC and PR-AUC
- Non-parametric bootstrap 95% Confidence Intervals (B=1000)
- Confusion Matrices and Curve plots
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, recall_score,
    precision_score, f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import os
from typing import Dict, Any, Tuple, Optional


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute comprehensive clinical classification metrics for a binary triage classifier.
    Positive class (1) = Preterm delivery
    Negative class (0) = Term delivery
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    sens = float(recall_score(y_true, y_pred, zero_division=0))
    
    # Specificity: True Negative Rate = TN / (TN + FP)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = float(np.mean(y_true))

    return {
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'sensitivity_recall': sens,
        'specificity': spec,
        'precision': prec,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn)
    }


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = 1000,
    threshold: float = 0.5,
    random_state: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Compute 95% Confidence Intervals for key metrics using non-parametric percentile bootstrapping.
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    boot_metrics = {
        'accuracy': [], 'balanced_accuracy': [], 'sensitivity_recall': [],
        'specificity': [], 'precision': [], 'f1_score': [], 'roc_auc': [], 'pr_auc': []
    }

    for _ in range(n_bootstraps):
        boot_idx = rng.choice(n, size=n, replace=True)
        y_t_boot = y_true[boot_idx]
        y_p_boot = y_prob[boot_idx]

        # Ensure both classes present in bootstrap sample
        if len(np.unique(y_t_boot)) < 2:
            continue

        m = compute_metrics(y_t_boot, y_p_boot, threshold=threshold)
        for k in boot_metrics:
            boot_metrics[k].append(m[k])

    ci_results = {}
    for k, vals in boot_metrics.items():
        if len(vals) > 0:
            low = float(np.percentile(vals, 2.5))
            high = float(np.percentile(vals, 97.5))
            mean_val = float(np.mean(vals))
        else:
            low, high, mean_val = 0.0, 0.0, 0.0
        ci_results[k] = {'mean': mean_val, 'ci_lower': low, 'ci_upper': high}

    return ci_results


def plot_roc_pr_curves(
    results_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: str = 'figures/roc_pr_curves.png'
):
    """
    Plot comparative ROC and Precision-Recall curves across models.
    results_dict: {model_name: (y_true, y_prob)}
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    for name, (y_true, y_prob) in results_dict.items():
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_val = roc_auc_score(y_true, y_prob)
            ax1.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})")
        except Exception:
            pass

        try:
            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            pr_val = average_precision_score(y_true, y_prob)
            ax2.plot(rec, prec, label=f"{name} (PR-AUC = {pr_val:.3f})")
        except Exception:
            pass

    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Chance')
    ax1.set_xlabel('False Positive Rate (1 - Specificity)')
    ax1.set_ylabel('True Positive Rate (Sensitivity)')
    ax1.set_title('Receiver Operating Characteristic (ROC)')
    ax1.legend(loc='lower right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    baseline_prev = np.mean([v[0].mean() for v in results_dict.values()])
    ax2.axhline(baseline_prev, color='k', linestyle='--', alpha=0.6, label=f'Preterm Prevalence ({baseline_prev:.1%})')
    ax2.set_xlabel('Recall (Sensitivity)')
    ax2.set_ylabel('Precision (PPV)')
    ax2.set_title('Precision-Recall (PR) Curve')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: str
):
    """Save confusion matrix plot."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    fig, ax = plt.subplots(figsize=(4.5, 4), dpi=150)
    cax = ax.matshow(cm, cmap='Blues')
    fig.colorbar(cax)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), va='center', ha='center', fontsize=14,
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Term (0)', 'Preterm (1)'])
    ax.set_yticklabels(['Term (0)', 'Preterm (1)'])
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Clinical Outcome')
    ax.set_title(f'Confusion Matrix: {model_name}', pad=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
