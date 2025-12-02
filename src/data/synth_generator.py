#!/usr/bin/env python3
"""Synthetic microbiome + clinical dataset generator for demo purposes.

Generates:
- taxa_abundance.csv (patient x taxa)
- metadata.csv (patient meta + ae_label)
"""
import os, argparse, numpy as np, pandas as pd
from scipy.stats import bernoulli


def generate_taxa(n_patients, n_taxa, ae_rate, seed=42):
    np.random.seed(seed)
    base_alpha = np.random.uniform(0.5, 3.0, size=n_patients)
    taxa = []
    for i in range(n_patients):
        x = np.random.gamma(shape=1.0 + base_alpha[i], scale=1.0, size=n_taxa)
        x = x / x.sum()
        taxa.append(x)
    taxa = np.stack(taxa)
    risky_idx = np.arange(0, min(10, n_taxa))
    risk_score = (0.5 - taxa[:, :len(risky_idx)].sum(axis=1)) + (1.0 - base_alpha / 3.0)
    prob = 1 / (1 + np.exp(- ( -1.5 + 3.0 * (risk_score - risk_score.mean()) / (risk_score.std()+1e-9) ) ) )
    labels = bernoulli.rvs(prob * 0.5 + ae_rate * 0.5)
    return taxa, labels, risky_idx


def generate_metadata(n_patients, ae_labels, seed=43):
    np.random.seed(seed)
    ages = np.random.randint(30, 85, size=n_patients)
    sex = np.random.choice(["M","F"], size=n_patients)
    bmi = np.round(np.random.normal(26, 4, size=n_patients),1)
    antibiotics_recent = np.random.binomial(1, 0.15, size=n_patients)
    treatment = np.random.choice(["immunotherapy","chemotherapy","targeted"], size=n_patients, p=[0.4,0.4,0.2])
    rows = []
    for i in range(n_patients):
        rows.append({
            "patient_id": f"P{i:04d}",
            "age": int(ages[i]),
            "sex": sex[i],
            "bmi": float(bmi[i]),
            "recent_antibiotic": int(antibiotics_recent[i]),
            "treatment_type": treatment[i],
            "ae_label": int(ae_labels[i])
        })
    return pd.DataFrame(rows)


def save_output(out, taxa, taxa_cols, meta):
    os.makedirs(out, exist_ok=True)
    taxa_df = pd.DataFrame(taxa, columns=taxa_cols)
    taxa_df.insert(0, "patient_id", meta["patient_id"].values)
    taxa_df.to_csv(os.path.join(out, "taxa_abundance.csv"), index=False)
    meta.to_csv(os.path.join(out, "metadata.csv"), index=False)
    print("Saved synthetic data to", out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/synthetic", help="output folder")
    parser.add_argument("--n_patients", type=int, default=200)
    parser.add_argument("--n_taxa", type=int, default=100)
    parser.add_argument("--ae_rate", type=float, default=0.12)
    args = parser.parse_args()
    taxa, labels, risky_idx = generate_taxa(args.n_patients, args.n_taxa, args.ae_rate)
    taxa_cols = [f"taxa_{i}" for i in range(taxa.shape[1])]
    meta = generate_metadata(args.n_patients, labels)
    save_output(args.out, taxa, taxa_cols, meta)


if __name__ == "__main__":
    main()

