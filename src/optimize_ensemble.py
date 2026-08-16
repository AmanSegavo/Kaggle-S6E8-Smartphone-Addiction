"""
Ensemble Blender and Weight Optimizer
Finds optimal weights for LightGBM, XGBoost, and CatBoost out-of-fold predictions.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize


def optimize_weights(y_true, predictions_dict: dict):
    """
    Optimize weights for multiple model predictions to maximize ROC-AUC.
    """
    model_names = list(predictions_dict.keys())
    preds_matrix = np.column_stack([predictions_dict[k] for k in model_names])
    n_models = len(model_names)

    def loss(weights):
        # Normalize weights to sum to 1
        w = weights / np.sum(weights)
        blended = np.dot(preds_matrix, w)
        return -roc_auc_score(y_true, blended)

    init_weights = np.ones(n_models) / n_models
    bounds = [(0, 1) for _ in range(n_models)]
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

    res = minimize(loss, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    opt_weights = res.x / np.sum(res.x)

    print("=" * 60)
    print("🎯 OPTIMIZED ENSEMBLE WEIGHTS:")
    print("=" * 60)
    for name, w in zip(model_names, opt_weights):
        print(f"  • {name:15s} : {w:.4f} ({w*100:.2f}%)")

    best_auc = -res.fun
    print(f"\n🏆 Optimal Blended ROC-AUC : {best_auc:.6f}")
    print("=" * 60)
    return dict(zip(model_names, opt_weights)), best_auc
