"""
explainability.py - Feature Importance and Electrophysiological Interpretability Analysis

Computes:
1. Standardized Logistic Regression Odds Ratios and Coefficients
2. Random Forest Gini Feature Importances
3. Visualizes top 15 electrophysiological biomarker drivers
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_feature_importances(
    base_dir: str = r'c:\Users\piyus\Downloads\ML 2\term-preterm-ehg-database-1.0.1\ehg_triage'
):
    models_dir = os.path.join(base_dir, 'models')
    reports_dir = os.path.join(base_dir, 'reports')
    figures_dir = os.path.join(base_dir, 'figures')
    data_dir = os.path.join(base_dir, 'data')

    model_path = os.path.join(models_dir, 'best_triage_model.joblib')
    features_path = os.path.join(models_dir, 'feature_names.json')
    features_csv = os.path.join(data_dir, 'features_60s_center.csv')

    model = joblib.load(model_path)
    with open(features_path, 'r') as f:
        feature_names = json.load(f)

    # 1. Logistic Regression Coefficients
    if hasattr(model, 'coef_'):
        coefs = model.coef_[0]
        odds_ratios = np.exp(coefs)
        df_imp = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefs,
            'Abs_Coefficient': np.abs(coefs),
            'Odds_Ratio': odds_ratios
        }).sort_values(by='Abs_Coefficient', ascending=False).reset_index(drop=True)
    else:
        df_imp = pd.DataFrame({'Feature': feature_names})

    # 2. Also fit Random Forest to get tree importances
    df_data = pd.read_csv(features_csv)
    df_data = df_data[df_data['source'] == 'TPEHGDB']
    X = df_data[feature_names].values
    y = df_data['label'].values

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)
    rf.fit(X, y)
    df_imp['RF_Importance'] = [rf.feature_importances_[feature_names.index(f)] for f in df_imp['Feature']]

    df_imp.to_csv(os.path.join(reports_dir, 'feature_importance.csv'), index=False)
    print("Top 10 Feature Drivers:")
    print(df_imp[['Feature', 'Coefficient', 'Odds_Ratio', 'RF_Importance']].head(10).to_string(index=False))

    # 3. Plot Top 15 Features
    top15 = df_imp.head(15)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    y_pos = np.arange(len(top15))
    colors = ['#d9534f' if c > 0 else '#428bca' for c in top15['Coefficient']]
    ax.barh(y_pos, top15['Coefficient'], color=colors, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top15['Feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Standardized Logistic Regression Coefficient (Log-Odds)')
    ax.set_title('Top 15 Electrophysiological Predictors of Preterm Delivery Risk\n(Red: Increases Preterm Risk | Blue: Protective / Term Associated)')
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'feature_importance.png'))
    plt.close()
    print(f"Feature importance figure saved to {os.path.join(figures_dir, 'feature_importance.png')}")


if __name__ == '__main__':
    compute_feature_importances()
