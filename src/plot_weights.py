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

    history_path = os.path.join(MODELS_DIR, "weights_history.txt")
    weights_history = []
    with open(history_path, "r") as f:
        for line in f:
            weights_history.append(list(map(float, line.strip().split(","))))

    weights_history = np.array(weights_history)
    epochs = np.arange(len(weights_history))

    num_features = weights_history.shape[1]

    plt.figure()
    for i in range(num_features):
        plt.plot(epochs, weights_history[:, i], label=f"w{i}")

    plt.xlabel("Epoch")
    plt.ylabel("Weight Value")
    plt.title("Weight Convergence During Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()

    return PLOT_PATH
