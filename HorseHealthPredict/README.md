# Predict Health Outcomes of Horses

Proyek klasifikasi multi-kelas untuk memprediksi kondisi akhir seekor kuda (`died`, `euthanized`, atau `lived`) berdasarkan data klinis (suhu tubuh, denyut nadi, kondisi organ pencernaan, dll).

**Sumber dataset:** [Predict Health Outcomes of Horses – Kaggle](https://www.kaggle.com/competitions/classification-predict-health-outcomes-of-horses).

---

## Overview

Data ini merupakan data rekam medis kuda dengan gejala kolik (sakit perut), berisi ±28 fitur klinis (suhu rektal, denyut nadi, laju napas, kondisi selaput lendir, hasil pemeriksaan rektal, dll) untuk 1.235 baris data train dan 824 baris data test. Tantangan utama dataset ini:

- **Banyak missing value** di fitur-fitur pemeriksaan fisik, karena tidak semua pemeriksaan dilakukan pada setiap kasus.
- **Kelas target tidak seimbang**.
- **Fitur kategorikal bersifat ordinal secara medis** , sehingga butuh penanganan khusus agar informasinya tidak hilang.
- **Fitur `lesion` berupa kode medis multi-digit** yang perlu diurai dulu sebelum bisa dipakai model.

### Ringkasan Pipeline
1. **Eksplorasi & pengecekan missing value** di data train maupun test.
2. **Feature engineering** 
3. **Custom mapper** : mengubah fitur kategorikal yang sebenarnya ordinal menjadi skala numerik berurutan sesuai tingkat keparahan medisnya 
4. **Pipeline preprocessing** yang dirangkai dalam satu alur yang konsisten dipakai untuk data train, validasi, maupun test.
5. **Perbandingan beberapa algoritma** : Naive Bayes, KNN, Decision Tree, Random Forest, XGBoost.
6. **Hyperparameter tuning** dengan Optuna untuk dua kandidat model terbaik: **XGBoost** dan **CatBoost**.
7. **Evaluasi** dengan classification report (precision, recall, F1-score) di data train vs validasi.

---

## Hasil

| Model | Akurasi Train | Akurasi Validasi |
|---|---|---|
| XGBoost + tuning Optuna + SMOTE (pipeline penuh) | ~82.9% | **~71.7%** |
| CatBoost + tuning Optuna (subset 15 fitur terpilih) | ~75.2% | ~66.4% |

Model terbaik sejauh ini adalah **XGBoost** dengan pipeline lengkap (feature engineering + custom mapper + SMOTE + tuning Optuna), mencapai akurasi validasi sekitar **71–72%**.

---

## Kesimpulan

1. Skor terbaik yang berhasil dicapai sejauh ini **masih di kisaran 70%**, dan belum berhasil ditingkatkan lebih jauh meski sudah mencoba berbagai kombinasi.
2. Ada **gap yang cukup besar antara akurasi train dan validasi**, yang mengindikasikan model masih agak overfit terhadap data latih, meski sudah pakai regularisasi dan tuning.
3. Proyek ini **cukup menantang** dimana proyek ini memiliki **5 versi notebook berbeda** yang mencoba berbagai pendekatan (feature engineering, pemilihan model, teknik penyeimbangan kelas, seleksi fitur, dll), namun hasil akhirnya belum jauh berbeda secara signifikan dari satu versi ke versi lain. Ini menandakan bahwa peningkatan performa lebih lanjut kemungkinan butuh pendekatan yang lebih berbeda dari yang sudah dicoba, bukan sekadar tuning ulang parameter yang sama.

## Saran

Oleh Karena itu, dari kesimpulan yang didapat, perlu pengembangan lebih lanjut agar model lebih robust seperti mencoba model lain atau teknik ensemble lainnya, feature engineering yang lebih baik, serta uji kombinasi hyperparameter yang lebih variatif. saya akan sangat mengapresiasi jika ada saran dari pembaca
