import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error


def train_model(df):
    df = df.copy()

    # Assume last column is target
    target_col = df.columns[-1]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode categorical features
    X = pd.get_dummies(X)

    # Detect problem type
    if y.dtype == 'object' or y.nunique() < 10:
        problem_type = "classification"
        model = RandomForestClassifier()
    else:
        problem_type = "regression"
        model = RandomForestRegressor()

    # Encode target if needed
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        le = None

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Evaluation
    if problem_type == "classification":
        score = accuracy_score(y_test, predictions)
    else:
        score = mean_squared_error(y_test, predictions)

    # Feature importance
    importances = model.feature_importances_
    feature_names = X.columns

    feature_importance = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    return {
        "model": model,
        "type": problem_type,
        "score": round(score, 4),
        "target": target_col,
        "features": list(X.columns),
        "importance": feature_importance,
        "encoder": le
    }