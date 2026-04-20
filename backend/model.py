import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def load_model():

    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "..", "processed.cleveland.data")

    df = pd.read_csv(file_path, header=None, na_values="?")

    df.columns = [
        "age","sex","cp","trestbps","chol",
        "fbs","restecg","thalach","exang",
        "oldpeak","slope","ca","thal","target"
    ]

    df = df.dropna()

    # 🔥 MEDIAN (ortalama yerine daha doğru)
    mean_values = df.median()

    X = df.drop("target", axis=1)
    y = df["target"].apply(lambda x: 1 if x > 0 else 0)

    # 🔥 SCALER
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 🔥 GELİŞMİŞ LOGISTIC REGRESSION
    model = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        C=1.5
    )

    model.fit(X, y)

    return model, mean_values, scaler