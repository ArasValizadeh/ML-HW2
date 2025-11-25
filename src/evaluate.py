from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import json
import os

def evaluate(model, X_test, y_test, save_path="results/metrics.json"):
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)

    results = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

    return results