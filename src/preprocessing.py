import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_clean(path: str):
    df = pd.read_csv(path)

    df = df.dropna()
    df = df.drop(columns=["id"])

    df["sex"] = df["sex"].map({"M": 1, "F": 0})

    df["fracture"] = df["fracture"].map({"fracture": 1, "no fracture": 0})

    df = pd.get_dummies(df, columns=["medication"], prefix="med")

    df["BMI"] = df["weight_kg"] / (df["height_cm"] / 100) ** 2

    numeric_cols = ["age", "weight_kg", "height_cm", "waiting_time", "BMI"]
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])



    return df
