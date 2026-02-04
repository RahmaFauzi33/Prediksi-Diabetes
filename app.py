# app.py
"""
Streamlit app untuk prediksi diabetes (LogReg / RandomForest)
AMAN untuk Pipeline + ColumnTransformer + OneHotEncoder
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ==============================
# CONFIG
# ==============================
st.set_page_config(
    page_title="Prediksi Diabetes",
    page_icon="🩺",
    layout="centered",
)


# ==============================
# MODEL LOADING
# ==============================
def resolve_model_path() -> str:
    env_path = os.environ.get("MODEL_PATH")
    if env_path:
        return env_path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "model.joblib")


MODEL_PATH = resolve_model_path()


@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"model.joblib tidak ditemukan di:\n{path}\n\n"
            "Pastikan struktur:\n"
            "project/\n"
            " ├─ app.py\n"
            " └─ model.joblib"
        )
    return joblib.load(path)


# ==============================
# FIXED FEATURE LIST (ANTI ERROR)
# ==============================
def get_expected_features() -> List[str]:
    """
    HARUS SESUAI fitur mentah dataset training
    (sebelum OneHotEncoder)
    """
    return [
        "gender",
        "age",
        "hypertension",
        "heart_disease",
        "smoking_history",
        "bmi",
        "HbA1c_level",
        "blood_glucose_level",
    ]


# ==============================
# PREDICTION
# ==============================
def predict_probability(model, X: pd.DataFrame) -> float:
    """Probabilitas diabetes (kelas 1)"""
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(X)[0][1])

    # fallback (jarang kepakai)
    score = float(model.decision_function(X)[0])
    return float(1 / (1 + np.exp(-score)))


# ==============================
# INPUT WIDGETS
# ==============================
def widget_for_feature(name: str):
    label = name.replace("_", " ").title()

    if name == "gender":
        return st.selectbox(label, ["Female", "Male"], index=0)

    if name == "smoking_history":
        return st.selectbox(
            label,
            ["never", "No Info", "current", "former", "ever", "not current"],
            index=0,
        )

    if name in {"hypertension", "heart_disease"}:
        return st.selectbox(label, [0, 1], index=0)

    if name == "age":
        return st.number_input(label, 0, 120, 45, step=1)

    if name == "bmi":
        return st.number_input(label, 0.0, 80.0, 27.5, step=0.1)

    if name == "HbA1c_level":
        return st.number_input(label, 0.0, 20.0, 6.2, step=0.1)

    if name == "blood_glucose_level":
        return st.number_input(label, 0.0, 500.0, 140.0, step=1.0)

    return st.text_input(label, "")


# ==============================
# MAIN APP
# ==============================
def main():
    st.warning("VERSI STABIL AKTIF")

    st.title("🩺 Prediksi Diabetes")
    st.caption("Aplikasi Streamlit – Machine Learning (LogReg / RandomForest)")

    # Sidebar
    with st.sidebar:
        st.subheader("Pengaturan")
        threshold = st.slider(
            "Threshold klasifikasi",
            min_value=0.05,
            max_value=0.95,
            value=0.50,
            step=0.01,
        )
        st.markdown(
            "- Threshold bisa diubah untuk menyeimbangkan precision / recall\n"
            "- Dataset bersifat imbalanced"
        )

    # Load model
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        st.error(str(e))
        st.stop()

    features = get_expected_features()

    # Input form
    st.subheader("Input Data Pasien")
    with st.form("input_form"):
        inputs: Dict[str, Any] = {}
        cols = st.columns(2)

        for i, feat in enumerate(features):
            with cols[i % 2]:
                inputs[feat] = widget_for_feature(feat)

        submitted = st.form_submit_button("Prediksi")

    # Prediction
    if submitted:
        X = pd.DataFrame([inputs], columns=features)

        try:
            proba = predict_probability(model, X)
            pred = int(proba >= threshold)
        except Exception as e:
            st.error("Gagal melakukan prediksi")
            st.exception(e)
            st.stop()

        # Output
        st.subheader("Hasil Prediksi")
        c1, c2 = st.columns(2)

        with c1:
            st.metric("Kelas", "Diabetes (1)" if pred else "Non-Diabetes (0)")

        with c2:
            st.metric("Probabilitas", f"{proba*100:.2f}%")

        st.progress(min(max(proba, 0.0), 1.0))
        st.caption(f"Threshold: {threshold:.2f}")

        with st.expander("Debug – Input Data"):
            st.json(inputs)

        st.info(
            "Catatan: Aplikasi ini bersifat akademik dan **bukan** alat diagnosis medis."
        )


if __name__ == "__main__":
    main()
