import argparse, os
from src.preprocess.feature_builder import load_data, build_features
from src.models.xgb_model import train_xgb, save_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/synthetic")
    parser.add_argument("--out", type=str, default="outputs/xgb_model.pkl")
    args = parser.parse_args()
    taxa, meta = load_data(args.data_dir)
    X, feature_names, y, scaler = build_features(taxa, meta)
    model = train_xgb(X, y)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_model(model, args.out)
    print("Model trained and saved to", args.out)


if __name__=="__main__":
    main()

