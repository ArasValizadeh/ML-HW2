from sklearn.linear_model import LinearRegression
import joblib
import os

MODELS_DIR = os.path.join(os.path.dirname(__file__), "../models")

MODEL_PATH = os.path.join(MODELS_DIR, "linear_regression.pkl")

def train_and_save(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)

    return model, model.coef_, MODEL_PATH