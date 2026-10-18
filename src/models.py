"""
models.py - Model Architectures and Training Framework for EHG Preterm Risk Triage

Models Implemented:
1. Majority-Class Baseline (DummyClassifier)
2. Logistic Regression (L2 regularized, balanced class weights)
3. Support Vector Machine (RBF kernel, balanced class weights, Platt scaling)
4. Random Forest / XGBoost (ensemble of decision trees, balanced weights)
5. 1D CNN (PyTorch, lightweight 3-channel time-series convolutional network)
6. CNN-plus-Engineered-Feature Fusion Model (PyTorch multi-modal architecture)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from typing import Dict, Any, Tuple, Optional, List


# -------------------------------------------------------------------------
# PyTorch 1D CNN Architecture
# -------------------------------------------------------------------------

class EHG1DCNN(nn.Module):
    """
    Lightweight, heavily regularized 1D Convolutional Neural Network for 60s 3-channel EHG.
    Input shape: (batch_size, 3, 1200)
    Output shape: (batch_size, 1) logits
    """
    def __init__(self, in_channels: int = 3, n_classes: int = 1, dropout: float = 0.4):
        super(EHG1DCNN, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, stride=2, padding=3),  # (B, 16, 600)
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2)  # (B, 16, 300)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),  # (B, 32, 300)
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)  # (B, 32, 150)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),  # (B, 64, 150)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # (B, 64, 1)
        )
        
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.75),
            nn.Linear(32, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 1200)
        feat1 = self.conv1(x)
        feat2 = self.conv2(feat1)
        feat3 = self.conv3(feat2)
        flat = feat3.view(feat3.size(0), -1)
        out = self.fc(flat)
        return out


class EHGFusionModel(nn.Module):
    """
    Multi-modal fusion model combining 1D CNN signal representations with engineered tabular features.
    """
    def __init__(self, in_channels: int = 3, n_tabular_features: int = 35, dropout: float = 0.4):
        super(EHGFusionModel, self).__init__()
        
        # CNN backbone
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.cnn_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64, 24),
            nn.ReLU()
        )
        
        # Tabular MLP branch
        self.tab_proj = nn.Sequential(
            nn.Linear(n_tabular_features, 24),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Fusion classification head
        self.head = nn.Sequential(
            nn.Linear(24 + 24, 24),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(24, 1)
        )

    def forward(self, x_sig: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
        f1 = self.conv1(x_sig)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        cnn_rep = self.cnn_proj(f3.view(f3.size(0), -1))
        
        tab_rep = self.tab_proj(x_tab)
        fused = torch.cat([cnn_rep, tab_rep], dim=1)
        return self.head(fused)


# -------------------------------------------------------------------------
# PyTorch Dataset & Training Helpers
# -------------------------------------------------------------------------

class EHGDataset(Dataset):
    def __init__(self, signals: np.ndarray, labels: np.ndarray, tab_features: Optional[np.ndarray] = None):
        # signals: (N, 1200, 3) -> transpose to (N, 3, 1200)
        self.signals = torch.tensor(signals, dtype=torch.float32).permute(0, 2, 1)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        self.tab_features = torch.tensor(tab_features, dtype=torch.float32) if tab_features is not None else None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.tab_features is not None:
            return self.signals[idx], self.tab_features[idx], self.labels[idx]
        return self.signals[idx], self.labels[idx]


def train_cnn_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    is_fusion: bool = False,
    device: str = 'cpu'
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        if is_fusion:
            x_sig, x_tab, y = batch
            x_sig, x_tab, y = x_sig.to(device), x_tab.to(device), y.to(device)
            out = model(x_sig, x_tab)
        else:
            x_sig, y = batch
            x_sig, y = x_sig.to(device), y.to(device)
            out = model(x_sig)
            
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
    return total_loss / len(loader.dataset)


def predict_cnn(
    model: nn.Module,
    loader: DataLoader,
    is_fusion: bool = False,
    device: str = 'cpu'
) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            if is_fusion:
                x_sig, x_tab, _ = batch
                x_sig, x_tab = x_sig.to(device), x_tab.to(device)
                logits = model(x_sig, x_tab)
            else:
                x_sig, _ = batch
                x_sig = x_sig.to(device)
                logits = model(x_sig)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.append(probs)
    return np.vstack(preds).flatten()


# -------------------------------------------------------------------------
# Sklearn Model Builders
# -------------------------------------------------------------------------

def build_model(model_name: str, random_state: int = 42) -> Any:
    """Factory function for classical ML models."""
    if model_name == 'majority':
        return DummyClassifier(strategy='most_frequent')
    elif model_name == 'logistic_regression':
        return LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=random_state)
    elif model_name == 'svm_rbf':
        return SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', probability=True, random_state=random_state)
    elif model_name == 'random_forest':
        return RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=random_state)
    elif model_name == 'xgboost':
        # scale_pos_weight approximately 6.0 for ~13% preterm prevalence
        return XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.05, scale_pos_weight=6.0, eval_metric='logloss', random_state=random_state)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
