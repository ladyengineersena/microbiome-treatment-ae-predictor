import streamlit as st
import pandas as pd
import os
import sys
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess.feature_builder import load_data, build_features
from src.models.xgb_model import load_model, predict_xgb
from src.explain.shap_explain import explain_model, get_top_features

st.title("Microbiome AE Risk Predictor (Research Demo)")
st.markdown("**⚠️ This is a research prototype. Not for clinical use.**")
st.markdown("This demo uses synthetic data only. No real patient data is shown.")


data_dir = "data/synthetic"
model_path = "outputs/xgb_model.pkl"

if not os.path.exists(data_dir):
    st.warning("⚠️ No synthetic data found. Please run `python src/data/synth_generator.py` first.")
    st.stop()

if not os.path.exists(model_path):
    st.warning("⚠️ No trained model found. Please run `python src/train.py` first.")
    st.stop()

# Load data
try:
    taxa, meta = load_data(data_dir)
    X, feature_names, y, scaler = build_features(taxa, meta)
    model = load_model(model_path)
    
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox("Choose page", ["Patient Risk Assessment", "Dataset Overview", "Model Performance"])
    
    if page == "Dataset Overview":
        st.header("Dataset Overview")
        st.write(f"Total patients: {len(meta)}")
        st.write(f"Patients with AE: {y.sum()} ({y.mean()*100:.1f}%)")
        st.write("\n### Sample Metadata (first 10 patients):")
        st.dataframe(meta.head(10))
        
        st.write("\n### Treatment Distribution:")
        st.bar_chart(meta["treatment_type"].value_counts())
        
    elif page == "Model Performance":
        st.header("Model Performance")
        from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
        
        preds = predict_xgb(model, X)
        auc = roc_auc_score(y, preds)
        st.metric("AUC-ROC", f"{auc:.3f}")
        
        pred_binary = (preds > 0.5).astype(int)
        st.write("### Classification Report:")
        st.text(classification_report(y, pred_binary))
        
        cm = confusion_matrix(y, pred_binary)
        st.write("### Confusion Matrix:")
        st.write(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
        st.write(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")
        
    elif page == "Patient Risk Assessment":
        st.header("Patient Risk Assessment")
        
        patient_ids = meta["patient_id"].tolist()
        selected_pid = st.selectbox("Select Patient ID", patient_ids[:50])
        
        if st.button("Generate Risk Report"):
            patient_idx = patient_ids.index(selected_pid)
            patient_row = meta.iloc[patient_idx]
            patient_features = X[patient_idx:patient_idx+1]
            
            # Prediction
            risk_prob = predict_xgb(model, patient_features)[0]
            
            st.subheader(f"Patient: {selected_pid}")
            st.write("### Clinical Information:")
            st.json({
                "Age": int(patient_row["age"]),
                "Sex": patient_row["sex"],
                "BMI": float(patient_row["bmi"]),
                "Treatment Type": patient_row["treatment_type"],
                "Recent Antibiotic": bool(patient_row["recent_antibiotic"]),
                "Actual AE Status": "Yes" if patient_row["ae_label"] == 1 else "No"
            })
            
            st.write("### Risk Assessment:")
            risk_color = "🔴" if risk_prob > 0.5 else "🟡" if risk_prob > 0.3 else "🟢"
            st.metric("Predicted AE Risk Probability", f"{risk_prob:.3f}", delta=None)
            st.write(f"**Risk Level:** {risk_color} {'High' if risk_prob > 0.5 else 'Moderate' if risk_prob > 0.3 else 'Low'}")
            
            # SHAP explanation
            st.write("### Feature Importance (SHAP values):")
            shap_df, explainer = explain_model(model, patient_features, feature_names)
            top_features = get_top_features(shap_df, n_top=10)
            
            st.write("**Top contributing features for this patient:**")
            for feat, val in top_features.items():
                shap_val = shap_df[feat].iloc[0]
                direction = "↑ Increases risk" if shap_val > 0 else "↓ Decreases risk"
                st.write(f"- **{feat}**: {shap_val:.4f} ({direction})")
            
            st.info("💡 **Note:** This is a research prototype. Clinical decisions must not be based on these predictions.")
            
except Exception as e:
    st.error(f"Error loading data or model: {str(e)}")
    st.write("Please ensure you have:")
    st.write("1. Generated synthetic data: `python src/data/synth_generator.py`")
    st.write("2. Trained the model: `python src/train.py`")

