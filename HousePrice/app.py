from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib


BASE_DIR = Path(__file__).parent

@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE_DIR / "gbr_model.joblib")
    encoders = joblib.load(BASE_DIR / "train_encoders.joblib")
    scaler = joblib.load(BASE_DIR / "scaler_train.joblib")
    return model, encoders, scaler


# =====================================
# 1. KONFIGURASI APLIKASI
# =====================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# =====================================
# 2. LOAD MODEL DAN PREPROCESSOR
# =====================================

@st.cache_resource
def load_artifacts():
    model = joblib.load("gbr_model.joblib")
    encoders = joblib.load("train_encoders.joblib")
    scaler = joblib.load("scaler_train.joblib")
    return model, encoders, scaler

model, encoders, scaler = load_artifacts()

FEATURES = list(model.feature_names_in_)
SCALER_FEATURES = list(scaler.feature_names_in_)

# =====================================
# 3. TAMPILAN APLIKASI
# =====================================

st.title("🏠 House Price Prediction")

st.write(
    "Aplikasi untuk memperkirakan harga jual rumah "
    "berdasarkan karakteristik properti."
)

st.info(
    f"Model menggunakan {len(FEATURES)} fitur "
    "untuk menghasilkan prediksi."
)

st.subheader("Input Data Rumah")

input_data = {}

# Membagi tampilan menjadi dua kolom
col1, col2 = st.columns(2)

for i, feature in enumerate(FEATURES):

    # Bergantian antara kolom kiri dan kanan
    container = col1 if i % 2 == 0 else col2

    with container:

        # Fitur kategorikal
        if feature in encoders:

            categories = encoders[feature].classes_.tolist()

            input_data[feature] = st.selectbox(
                feature,
                options=categories,
                key=feature
            )

        # Fitur numerik
        else:

            input_data[feature] = st.number_input(
                feature,
                value=0.0,
                step=1.0,
                key=feature
            )

# =====================================
# 4. PREDIKSI
# =====================================

if st.button("Prediksi Harga Rumah", type="primary"):

    try:
        # Membuat DataFrame dari input
        df_input = pd.DataFrame([input_data])

        # Menyiapkan seluruh kolom yang dibutuhkan scaler
        df_scaler = pd.DataFrame(
            0.0,
            index=[0],
            columns=SCALER_FEATURES
        )

        # Menyiapkan data numerik untuk scaler
        for feature in SCALER_FEATURES:

            if feature in df_input.columns:
                if feature not in encoders:
                    df_scaler[feature] = pd.to_numeric(
                        df_input[feature]
                    )

        # Melakukan standardisasi
        scaled_values = scaler.transform(df_scaler)

        df_scaled = pd.DataFrame(
            scaled_values,
            columns=SCALER_FEATURES
        )

        # Mengembalikan nilai fitur kategorikal
        for feature, encoder in encoders.items():

            if feature in df_input.columns:
                df_scaled[feature] = encoder.transform(
                    df_input[feature]
                )

        # Mengambil tepat 69 fitur sesuai model
        X_predict = df_scaled[FEATURES]

        # Prediksi
        prediction = model.predict(X_predict)[0]

        # Menampilkan hasil
        st.success("Prediksi berhasil!")

        st.subheader("Estimasi Harga Rumah")

        st.metric(
            label="Predicted Sale Price",
            value=f"${prediction:,.2f}"
        )

        st.caption(
            "Hasil merupakan estimasi model, "
            "bukan harga jual pasti."
        )

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")