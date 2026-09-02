from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model dan alat pemrosesnya
model = joblib.load('gbr_model.joblib')
encoders = joblib.load('train_encoders.joblib')
scaler = joblib.load('scaler_train.joblib')

# Mengambil Fitur yg sudah di select
TOP_FEATURES = [
    'OverallQual', 'GrLivArea', '2ndFlrSF', 'BsmtFinSF1', 'YearBuilt', 
    '1stFlrSF', 'TotalBsmtSF', 'LotArea', 'GarageCars', 'GarageArea', 
    'KitchenQual', 'BsmtQual', 'Neighborhood', 'FireplaceQu', 'TotRmsAbvGrd'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        content = request.json
        raw_data = content['data']
        df_input = pd.DataFrame(raw_data)
        
        # Mengkapi 15 fitur utama jika ada yang tidak dikirim user
        for col in TOP_FEATURES:
            if col not in df_input.columns:
                df_input[col] = 0
        
        # memotong df_input di awal agar HANYA berisi 15 kolom feature yg di select
        df_input = df_input[TOP_FEATURES].copy()
        
        # PROSES TRANSFORM FITUR KATEGORIKAL
        for col in df_input.columns:
            if col in encoders:
                le = encoders[col]
                kategori_sah = set(le.classes_)
                nilai_aman = df_input[col].apply(lambda x: x if x in kategori_sah else le.classes_[0])
                df_input[col] = le.transform(nilai_aman)
            else:
                
                if df_input[col].dtype == 'object':
                    df_input[col] = 0

        kolom_wajib_scaler = list(scaler.feature_names_in_)
        df_penambal_scaler = pd.DataFrame(0, index=df_input.index, columns=kolom_wajib_scaler)
        
        for col in df_input.columns:
            if col in df_penambal_scaler.columns:
                df_penambal_scaler[col] = df_input[col]
        
        # Jalankan transformasi skala
        df_scaled_array = scaler.transform(df_penambal_scaler)
        df_scaled_full = pd.DataFrame(df_scaled_array, columns=kolom_wajib_scaler)
        
        # Kembalikan nilai fitur teks asli (encoded) 
        kolom_kategori_all = [col for col in encoders.keys()]
        for col in kolom_kategori_all:
            if col in df_input.columns:
                df_scaled_full[col] = df_input[col].values

        # Ambil daftar seluruh nama kolom pada saat training GBR
        kolom_wajib_gbr = list(model.feature_names_in_)
        
        # Buat DataFrame baru khusus untuk model GBR
        df_penambal_gbr = pd.DataFrame(0, index=df_input.index, columns=kolom_wajib_gbr)
        
        # Salin seluruh nilai fitur
        for col in df_penambal_gbr.columns:
            if col in df_scaled_full.columns:
                df_penambal_gbr[col] = df_scaled_full[col]
            elif col in df_input.columns:
                df_penambal_gbr[col] = df_input[col]

        # PREDIKSI AKHIR 
        predictions = model.predict(df_penambal_gbr)
        
        return jsonify({'predictions': predictions.tolist()})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(port=5000)
