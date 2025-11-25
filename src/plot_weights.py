import os
import joblib
import matplotlib.pyplot as plt
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(__file__), "../models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../results")
MODEL_PATH = os.path.join(MODELS_DIR, "linear_regression.pkl")
PLOT_PATH = os.path.join(RESULTS_DIR, "weight_plot.png")


def plot_weights(feature_names=None):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model = joblib.load(MODEL_PATH)
    coefs = model.coef_

    if feature_names is None:
        feature_names = [f"w{i}" for i in range(len(coefs))]

    indices = np.arange(len(coefs))

    plt.figure()
    plt.bar(indices, coefs)
    plt.xticks(indices, feature_names, rotation=45, ha="right")
    plt.tight_layout()
    plt.xlabel("Features")
    plt.ylabel("Weights")
    plt.title("Linear Regression Weights")

    plt.savefig(PLOT_PATH)
    plt.close()

    return PLOT_PATH
