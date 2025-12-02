# Microbiome–Cancer Treatment Adverse Event Predictor (Research Prototype)

**Status:** Research prototype — Not for clinical use.  
**License:** NO LICENSE — All Rights Reserved (do not reuse without permission).

## What

Predicts risk of treatment-related adverse events (AEs) in cancer patients from stool microbiome profiles (shotgun or 16S) + clinical + treatment data. This repository contains code, a synthetic data generator, baseline modeling notebooks, explainability demo and a Streamlit demo mockup.

## Quickstart (local)

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt

# Generate synthetic data (example)
python src/data/synth_generator.py --out data/synthetic --n_patients 300 --ae_rate 0.12

# Train baseline model
python src/train.py --data_dir data/synthetic --out outputs/xgb_model.pkl

# Evaluate
python src/evaluate.py --data_dir data/synthetic --model outputs/xgb_model.pkl

# Run Streamlit app
streamlit run app/streamlit_app.py
```

## Notes

- This repo does NOT include any real patient data. Use only with synthetic/demo data here unless you have IRB and data-sharing agreements.
- Model outputs are for research and demonstration only. Clinical decisions must not be based on these prototypes.
- All Rights Reserved: Do not reuse code or data without explicit permission from the project owner.

## Project Structure

```
microbiome-treatment-ae-predictor/
├── data/
│   └── synthetic/                 # (generated)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_models.ipynb
│   └── 04_explainability.ipynb
├── src/
│   ├── data/
│   │   └── synth_generator.py
│   ├── preprocess/
│   │   └── feature_builder.py
│   ├── models/
│   │   └── xgb_model.py
│   ├── train.py
│   ├── evaluate.py
│   └── explain/
│       └── shap_explain.py
├── scripts/
│   ├── run_all.sh
│   └── run_all.bat
├── app/
│   └── streamlit_app.py
├── outputs/
├── README.md
├── ETHICS.md
├── PILOT_PROTOCOL.md
├── NO_LICENSE.txt
└── requirements.txt
```

