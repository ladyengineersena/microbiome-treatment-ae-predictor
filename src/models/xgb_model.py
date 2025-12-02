import xgboost as xgb
import joblib


def train_xgb(X, y, params=None):
    if params is None:
        params = {"objective":"binary:logistic", "eval_metric":"logloss", "use_label_encoder":False}
    dtrain = xgb.DMatrix(X, label=y)
    model = xgb.train(params, dtrain, num_boost_round=100)
    return model


def predict_xgb(model, X):
    d = xgb.DMatrix(X)
    return model.predict(d)


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)

