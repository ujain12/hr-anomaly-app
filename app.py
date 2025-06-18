# app.py  – HR Anomaly Detection UI  (final with robust confusion matrix)
import streamlit as st
import pandas as pd
from anomaly_detection import load_model_and_predict, train_on_clean
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import os

# ──────────────────────────────────────────────────────────────────
st.set_page_config(page_title="HR Anomaly Detector", layout="wide")
st.title("🔍 HR Anomaly Detection")
MODELS = ["IsolationForest", "OneClassSVM", "LocalOutlierFactor"]
# ──────────────────────────────────────────────────────────────────

# ─── Sidebar ─────────────────────────────────────────────────────
model_choice = st.sidebar.selectbox("Choose model", MODELS)

if st.sidebar.button("Retrain on synthetic_clean.csv"):
    if os.path.exists("synthetic_ipps_a_employees_clean.csv"):
        clean_df = pd.read_csv("synthetic_ipps_a_employees_clean.csv")
        train_on_clean(clean_df)
        st.sidebar.success("✅ Models retrained & saved.")
    else:
        st.sidebar.error("synthetic_clean.csv not found.")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "• Upload any HR CSV to flag anomalies.\n"
    "• If the file contains an **`IsAnomaly`** column you’ll see metrics."
)

# ─── Main area ───────────────────────────────────────────────────
uploaded = st.file_uploader("Upload HR CSV file", type=["csv"])

if uploaded is None:
    st.info("⬆️  Upload a CSV to begin.")
    st.stop()

# Predict anomalies
df = pd.read_csv(uploaded)
df["AnomalyFlag"] = load_model_and_predict(df, model_choice)

# Show dataset
st.subheader("Flagged dataset")
st.dataframe(df, use_container_width=True)

st.info(
    f"Total rows: **{len(df):,}** &nbsp;&nbsp;•&nbsp;&nbsp; "
    f"Anomalies flagged: **{int(df['AnomalyFlag'].sum()):,}**"
)

# ─── Metrics if ground-truth present ─────────────────────────────
if "IsAnomaly" in df.columns:
    y_true = df["IsAnomaly"].astype(int)
    y_pred = df["AnomalyFlag"]

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    st.write("### Evaluation metrics")
    st.markdown(
        f"* **Accuracy:** `{acc:.3f}`\n"
        f"* **Precision:** `{prec:.3f}`\n"
        f"* **Recall:** `{rec:.3f}`\n"
        f"* **F1 score:** `{f1:.3f}`"
    )

    cm_df = pd.DataFrame(
        [[tp, fp], [fn, tn]],
        index=["Model → Anomaly", "Model → Normal"],
        columns=["Actual → Anomaly", "Actual → Normal"],
        dtype=int
    )

    st.write("#### Confusion matrix (counts)")
    st.table(
        cm_df.style.set_table_styles(
            [{"selector": "th", "props": [("text-align", "center")]}]
        )
    )

    st.caption(
        """
        * **Model → Anomaly / Actual → Anomaly**  →  **True Positives** ✔️  
        * **Model → Anomaly / Actual → Normal**  →  **False Positives** ❌  
        * **Model → Normal / Actual → Anomaly**  →  **False Negatives** ❌  
        * **Model → Normal / Actual → Normal**  →  **True Negatives** ✔️
        """
    )

# ─── Download anomalies only ─────────────────────────────────────
csv_bytes = df[df["AnomalyFlag"] == 1].to_csv(index=False).encode()
st.download_button(
    "Download anomalies as CSV",
    csv_bytes,
    file_name="anomalies_only.csv",
    mime="text/csv"
)
