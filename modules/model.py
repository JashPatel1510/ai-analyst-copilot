import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score


def train_model(df, target_col=None, problem_type="auto"):
    df = df.copy()

    # ----------------------------
    # 1. Validate target column
    # ----------------------------
    if target_col is None or target_col not in df.columns:
        raise ValueError("Invalid or missing target column")

    # ----------------------------
    # 2. Drop ID columns (IMPORTANT FIX)
    # ----------------------------
    df = df.drop(columns=[
        col for col in df.columns
        if "id" in col.lower() and col != target_col
    ], errors="ignore")

    # ----------------------------
    # 3. Split features / target
    # ----------------------------
    X = df.drop(columns=[target_col])
    y = df[target_col].copy()

    # One-hot encode features
    X = pd.get_dummies(X)

    # ----------------------------
    # 4. Model selection
    # ----------------------------
    le = None

    if problem_type == "classification":
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            n_jobs=1,
            random_state=42
        )

        if y.dtype == "object" or pd.api.types.is_string_dtype(y):
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y))

    elif problem_type == "regression":
        model = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            n_jobs=1,
            random_state=42
        )

    else:
        # AUTO DETECT
        if y.dtype == "object" or y.nunique() < 10:
            model = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                n_jobs=1,
                random_state=42
            )

            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y))

            problem_type = "classification"

        else:
            model = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                n_jobs=1,
                random_state=42
            )

            problem_type = "regression"


    # ----------------------------
    # 5. Sample if dataset too large (Render free tier RAM limit)
    # ----------------------------
    if len(X) > 5000:
        X = X.sample(n=5000, random_state=42)
        y = y.loc[X.index]



    # ----------------------------
    # 6. Train/test split
    # ----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ----------------------------
    # 7. Train model
    # ----------------------------
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # ----------------------------
    # 8. Metrics
    # ----------------------------
    if problem_type == "classification":
        score = accuracy_score(y_test, predictions)
        rmse = None
        r2 = None
    else:
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        score = rmse

    # ----------------------------
    # 9. Feature importance (normalized)
    # ----------------------------
    importances = model.feature_importances_
    total = np.sum(importances)

    feature_importance = sorted(
        [
            (feat, (imp / total) * 100)
            for feat, imp in zip(X.columns, importances)
        ],
        key=lambda x: x[1],
        reverse=True
    )[:5]

    # ----------------------------
    # 10. Return result
    # ----------------------------
    return {
        "model": model,
        "type": problem_type,
        "score": round(score, 4),
        "rmse": round(rmse, 4) if rmse is not None else None,
        "r2": round(r2, 4) if r2 is not None else None,
        "target": target_col,
        "features": list(X.columns),
        "importance": feature_importance,
        "encoder": le
    }