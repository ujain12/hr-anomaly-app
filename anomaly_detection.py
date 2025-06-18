"""
anomaly_detection.py  – FINAL
• train_on_clean(clean_df): fits pre-processor + 3 models on clean data
• load_model_and_predict(df, model_name): flags anomalies in any CSV
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.svm      import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.compose  import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute   import SimpleImputer


# ------------------------------------------------------------------
def _build_preprocessor(num_cols, cat_cols):
    """ColumnTransformer: impute+scale numeric, impute+one-hot categorical."""
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imp",   SimpleImputer(strategy="mean")),
                ("scale", StandardScaler())
            ]), num_cols),
            ("cat", Pipeline([
                ("imp",    SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols)
        ],
        remainder="drop"
    )


def _coerce_types(df, num_cols, cat_cols):
    """Numeric → float (strings→NaN); categoricals → str."""
    df = df.copy()
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    df[cat_cols] = df[cat_cols].astype(str)
    return df


# ------------------------------------------------------------------
def train_on_clean(clean_df: pd.DataFrame):
    """Fit pre-processor + IsolationForest / SVM / LOF on CLEAN data only."""
    num_cols = clean_df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = clean_df.select_dtypes(include="object").columns.tolist()

    pre = _build_preprocessor(num_cols, cat_cols)
    X_clean = pre.fit_transform(_coerce_types(clean_df[num_cols + cat_cols],
                                              num_cols, cat_cols))
    joblib.dump(pre, "preprocessor.pkl")

    models = {
        "IsolationForest":    IsolationForest(contamination=0.1, random_state=42),
        "OneClassSVM":        OneClassSVM(nu=0.1, kernel="rbf", gamma="scale"),
        "LocalOutlierFactor": LocalOutlierFactor(n_neighbors=20,
                                                 contamination=0.1, novelty=True)
    }
    for name, model in models.items():
        model.fit(X_clean)
        joblib.dump(model, f"{name}.pkl")

    return list(models.keys())


# ------------------------------------------------------------------
# ─── REPLACE this function in anomaly_detection.py ────────────────
def load_model_and_predict(df: pd.DataFrame, model_name: str):
    """
    Return 1 = anomaly, 0 = normal.
    Works even if numeric columns have stray strings; keeps DataFrame
    so ColumnTransformer can select by column names.
    """
    pre   = joblib.load("preprocessor.pkl")
    model = joblib.load(f"{model_name}.pkl")

    # Columns memorised by the pre-processor during training
    locked_cols = pre.feature_names_in_.tolist()

    # Ensure we have those columns (add missing as NaN)
    df_fixed = df.reindex(columns=locked_cols)
    # Split into numeric / categorical based on training info
    num_cols = pre.transformers_[0][2]
    cat_cols = pre.transformers_[1][2]

    # Coerce dtypes
    df_fixed[num_cols] = df_fixed[num_cols].apply(pd.to_numeric, errors="coerce")
    df_fixed[cat_cols] = df_fixed[cat_cols].astype(str)

    # Keep as DataFrame (preserves column names)
    X = pre.transform(df_fixed)     # no .to_numpy() here
    preds = model.predict(X)
    return np.where(preds == -1, 1, 0)
# ──────────────────────────────────────────────────────────────────

