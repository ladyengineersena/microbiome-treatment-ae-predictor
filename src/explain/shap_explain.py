import shap
import pandas as pd
import numpy as np


def explain_model(model, X, feature_names, max_display=10):
    """Generate SHAP explanations for XGBoost model."""
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]  # For binary classification, use positive class
    df = pd.DataFrame(shap_vals, columns=feature_names)
    return df, explainer


def get_top_features(shap_df, n_top=10):
    """Get top contributing features by mean absolute SHAP value."""
    mean_abs = shap_df.abs().mean().sort_values(ascending=False)
    return mean_abs.head(n_top)

