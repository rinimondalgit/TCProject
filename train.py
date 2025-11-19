import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import joblib


DATA_PATH = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_PATH = Path("models/churn_model.pkl")


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and clean the Telco Customer Churn dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Place it in the data/ folder.")

    df = pd.read_csv(path)

    # Fix TotalCharges (has blank spaces)
    df["TotalCharges"] = df["TotalCharges"].replace(" ", pd.NA)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])

    # Convert target: Yes/No -> 1/0
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # Drop customerID
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    return df


def main():
    df = load_data()
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Split (stratify ensures both classes appear in train & validation)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "bool"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    models = {
        "logreg": LogisticRegression(
            solver="liblinear",
            max_iter=1000
        ),
        "rf": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        ),
    }

    best_auc = 0
    best_model_name = None
    best_pipeline = None

    for name, model in models.items():
        clf = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )

        clf.fit(X_train, y_train)
        y_val_pred = clf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_val_pred)

        print(f"{name} AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_pipeline = clf
            best_model_name = name

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)

    print(f"\nBest model: {best_model_name}  AUC={best_auc:.4f}")
    print(f"Saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
