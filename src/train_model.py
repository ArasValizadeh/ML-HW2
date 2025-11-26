from sklearn.linear_model import SGDRegressor
import joblib
import os

MODELS_DIR = os.path.join(os.path.dirname(__file__), "../models")

MODEL_PATH = os.path.join(MODELS_DIR, "linear_regression.pkl")

def train_and_save(X_train, y_train):
    model = SGDRegressor(max_iter=5, learning_rate="invscaling", warm_start=True)
    weights_history = []

    for epoch in range(200):
        model.fit(X_train, y_train)
        weights_history.append(model.coef_.copy())

    history_path = os.path.join(MODELS_DIR, "weights_history.txt")
    with open(history_path, "w") as f:
        for w in weights_history:
            f.write(",".join(map(str, w)) + "\n")

    joblib.dump(model, MODEL_PATH)

    return model, model.coef_, MODEL_PATH