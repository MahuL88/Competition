# House Price Prediction

Kompetisi machine learning untuk memprediksi harga jual rumah (`SalePrice`) menggunakan dataset **House Prices - Advanced Regression Techniques** (Kaggle), dengan `train.csv` (1460 baris, 81 kolom) untuk melatih model dan `test.csv` (1459 baris, 80 kolom) untuk prediksi akhir. Model final di-deploy sebagai REST API sederhana menggunakan Flask.

**Stack:** `pandas`, `numpy`, `seaborn`/`matplotlib`, `scikit-learn`, `optuna` (hyperparameter tuning), `flask` (deployment), `joblib` (menyimpan model & preprocessing objects).

Sumber Data : [House Price](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
---

## 1. Data Collecting and Loading 

Install dan Import library

## 2. Data Cleaning and Transformation — Data Train

1. cek missing value
2. Imputasi "no facility"
3. Drop kolom ber-importance ~0 
4. Ordinal encoding manual
5. Split kolom `less`/`over` 
6. Imputasi sisa: median (numerik) & modus (kategorikal) 
7. Scaling
8. Cek duplikat 
9. Encoding

## 3. Data Cleaning and Transformation — Data Test

1. Cek info & missing value 
2. Simpan `id_test` sebelum diproses
3. lakukan clean dan transform  kembali seperti perlakuan pada train

## 4. Exploratory Data Analysis
- Grid histogram 77 variabel, distribusi 5 kolom terpilih (`OverallQual`, `YearBuilt`, `LotArea`, `SaleType`, `SaleCondition`) 
- Correlation heatmap seluruh variabel numerik + bar chart korelasi terhadap `SalePrice`

## 5. Training
- Feature selection: drop kolom berkorelasi rendah 
- Latih 3 model
- Evaluasi MAE/MSE/R² tiap model 

## 6. Hyperparameter Tuning (Optuna)

-  `objective()`dengan beberapa params
-  `study.optimize` sebanyak  200 trial 
-  Retrain `GBR_opt` dengan best params, evaluasi ulang
-  Simpan ke `gbr_model.joblib` 

## 7. Feature Importance

- Ambil `feature_importances_` dari `GBR_opt`, tampilkan fitur
- Visualisasi bar chart 
- Save Model dan Prediksi Data Test Asli

## 8. Deployment

1. Ambil 3 baris sampel dari `test.csv` 
2. Membuat flask app
3. Jalankan `app.py` di background, tes lewat `app.test_client()` 

---

## Hasil

### Perbandingan Model (sebelum tuning)
| Model | MAE | MSE | R² |
|---|---|---|---|
| Lars | 62,018.67 | 7.47 × 10⁹ | 0.0259 |
| Linear Regression | 21,248.80 | 1.17 × 10⁹ | 0.8478 |
| **Gradient Boosting Regressor** | **16,765.17** | **6.87 × 10⁸** | **0.9105** |

### Setelah Hyperparameter Tuning
- Parameter terbaik: `n_estimators=255`, `learning_rate=0.044`, `max_depth=6`, `min_samples_split=9`, `min_samples_leaf=2`, `subsample=0.850`
- R² rata-rata 5-fold CV saat pencarian: **0.8771**
- Setelah retrain & evaluasi : **MAE = 15,770**, **R² = 0.9130**

### Feature Importance (GBR setelah tuning)
| Fitur | Importance |
|---|---|
| `OverallQual` | 54.5% |
| `GrLivArea` | 11.3% |
| `2ndFlrSF` | 4.2% |
| `TotalBsmtSF` | 3.4% |
| `BsmtFinSF1` | 3.3% |

### Uji Deployment
Endpoint `/predict` (dengan hanya 15 fitur teratas) berhasil menerima 3 baris data sampel dan mengembalikan prediksi:
```
[95,148.85, 120,688.88, 160,871.05]
```

---

## Kesimpulan

1. **Gradient Boosting Regressor** menjadi model terbaik dibanding `Lars` dan `Linear Regression`
2. Perbaikan imputasi (`0`/`'None'` untuk kolom tanpa fasilitas) **terbukti berdampak nyata** 
3. Endpoint deployment berhasil diuji

## Saran
Proyek Kompetisi ini perlu pengembangan lebih lanjut mengingat skor yang didapat dan hasil prediksi model masih jauh dari harapan, oleh karena itu terdapat saran yang akan diuji kelak dan sangat menerima saran dari pembaca untuk meningkatkan performa model yang dibangun, saran yang didapat yaitu : 
1. Endpoint deployment sekarang hanya menerima 15 fitur teratas dan mengisi `0` untuk fitur lain yang tidak dikirim
2. **Transformasi target** (`np.log1p(SalePrice)`) masih belum dicoba 
3. **Tambahkan monitoring sungguhan** di endpoint (logging tiap request/prediksi, endpoint `/health`, versi model tercatat eksplisit misal `gbr_model_v2.joblib`)
