# Klasifikasi Rock-Paper-Scissors (Streamlit)

Aplikasi web sederhana untuk Tugas Kelompok 2 mata kuliah Artificial Intelligence (BINUS Online). Pengguna meng-upload gambar tangan, lalu model CNN menebak apakah gambar itu **rock**, **paper**, atau **scissors**.

- Model: transfer learning MobileNetV2 + hyperparameter tuning (Keras Tuner), input 224×224, akurasi test 100%
- Halaman menampilkan gambar, kelas prediksi, confidence, dan bar probabilitas ketiga kelas
- Jika confidence di bawah 70%, muncul peringatan bahwa model kurang yakin

## Isi repo

| File | Isi |
|---|---|
| `app.py` | Aplikasi Streamlit |
| `inference.py` | `load_artifacts()` dan `predict_image()` (dari notebook training) |
| `model_cnn.keras` | Model terlatih (normalisasi piksel sudah di dalam model) |
| `class_names.json`, `model_config.json` | Urutan label dan ukuran input |
| `samples/` | Contoh gambar untuk mencoba aplikasi |
| `test_app.py` | Tes prediksi dan tes aplikasi |

## Instalasi

Butuh Python 3.10–3.13.

```bash
git clone https://github.com/fahruluzi/rps-streamlit-app.git
cd rps-streamlit-app
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Menjalankan

```bash
streamlit run app.py
```

Browser terbuka di http://localhost:8501. Upload gambar (jpg/png), misalnya dari folder `samples/`.

## Tes

```bash
pytest
```
