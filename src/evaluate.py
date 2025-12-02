import argparse
from src.preprocess.feature_builder import load_data, build_features
from src.models.xgb_model import predict_xgb, load_model
from sklearn.metrics import roc_auc_score, classification_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/synthetic")
    parser.add_argument("--model", type=str, default="outputs/xgb_model.pkl")
    args = parser.parse_args()
    taxa, meta = load_data(args.data_dir)
    X, feature_names, y, scaler = build_features(taxa, meta)
    model = load_model(args.model)
    preds = predict_xgb(model, X)
    print("AUC:", roc_auc_score(y, preds))
    print(classification_report(y, preds>0.5))


if __name__=="__main__":
    main()

