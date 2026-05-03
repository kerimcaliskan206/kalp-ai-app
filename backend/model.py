import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

def load_model():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "..", "processed.cleveland.data")

    df = pd.read_csv(file_path, header=None, na_values="?")
    df.columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    ]
    df = df.dropna()
    mean_values = df.median()

    X = df.drop("target", axis=1)
    y = df["target"].apply(lambda x: 1 if x > 0 else 0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    rf_params = {'n_estimators': [100, 200], 'max_depth': [5, 10, None]}
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=3).fit(X_train, y_train)
    model_rf = rf_grid.best_estimator_

    xgb_params = {'learning_rate': [0.01, 0.1], 'max_depth': [3, 5], 'n_estimators': [100]}
    xgb_grid = GridSearchCV(XGBClassifier(eval_metric='logloss'), xgb_params, cv=3).fit(X_train, y_train)
    model_xgb = xgb_grid.best_estimator_

    model_lr = LogisticRegression(max_iter=5000).fit(X_train, y_train)

    preds = (model_rf.predict(X_test) + model_xgb.predict(X_test) + model_lr.predict(X_test)) / 3
    final_preds = [1 if p >= 0.5 else 0 for p in preds]
    accuracy = accuracy_score(y_test, final_preds) * 100

    return {"rf": model_rf, "xgb": model_xgb, "lr": model_lr}, mean_values, scaler, round(accuracy, 2)