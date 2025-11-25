from src.preprocessing import load_and_clean
from src.split_data import split_data
from src.train_model import train_and_save
from src.evaluate import evaluate
from src.plot_weights import plot_weights
from src.config import DATA_PATH

df = load_and_clean(DATA_PATH)

X_train, X_test, y_train, y_test = split_data(df)

model, weights, model_path = train_and_save(X_train, y_train)
print(f"Model saved at: {model_path}")

results = evaluate(model, X_test, y_test)
print("Evaluation results:", results)

plot_path = plot_weights(feature_names=X_train.columns)
print(f"Weight plot generated at: {plot_path}")