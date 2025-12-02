import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(data_dir):
    taxa = pd.read_csv(f"{data_dir}/taxa_abundance.csv")
    meta = pd.read_csv(f"{data_dir}/metadata.csv")
    return taxa, meta


def build_features(taxa_df, meta_df, n_top=50):
    taxa_only = taxa_df.drop(columns=["patient_id"]).values + 1e-9
    log_t = np.log(taxa_only)
    var_idx = np.argsort(log_t.var(axis=0))[::-1][:n_top]
    X_taxa = log_t[:, var_idx]
    taxa_cols = [f"taxa_{i}" for i in var_idx]
    X_tab = meta_df[["age","bmi"]].values
    X = np.concatenate([X_taxa, X_tab], axis=1)
    feature_names = taxa_cols + ["age","bmi"]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, feature_names, meta_df["ae_label"].values, scaler

