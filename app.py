from flask import Flask, render_template, request
import joblib #manggil file .pkl
import numpy as np

# 1. Inisialisasi Aplikasi Flask
app = Flask(__name__) #membuat server flask baru

# 2. Load Model dan Scaler
try:
    model = joblib.load('model_knn_stunting.pkl')
    scaler = joblib.load('scaler.pkl') # Memuat file scaler Anda
    print("Model dan Scaler berhasil dimuat.")
except Exception as e:
    print(f"Error saat memuat file: {e}")

# 3. Route Halaman Utama
@app.route('/')
def home():
    return render_template('index.html')

# 4. Route Proses Prediksi
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Mengambil data dari form
            umur = float(request.form['umur'])
            jk_input = request.form['jenis_kelamin']
            tinggi = float(request.form['tinggi'])

            # Konversi Jenis Kelamin (sesuaikan dengan training Anda)
            # laki-laki = 1, perempuan = 0
            jk = 1 if jk_input == 'laki-laki' else 0

            # Susun fitur dalam bentuk array 2D
            # Urutan HARUS SAMA dengan saat training: [Umur, Jenis Kelamin, Tinggi]
            fitur_raw = np.array([[umur, jk, tinggi]])
            		
            # --- LANGKAH KRUSIAL: SCALING ---
            # Mengubah data input user menjadi skala yang sama dengan data training
            fitur_scaled = scaler.transform(fitur_raw)
            
            # Melakukan prediksi dengan data yang sudah di-scale
            prediksi = model.predict(fitur_scaled)
            angka_hasil = int(prediksi[0])

            # Mapping Label (Sesuaikan urutan angka dengan LabelEncoder Anda)
            status_map = {
                0: "Normal",
                1: "Sangat Pendek",
                2: "Pendek",
                3: "Tinggi"
            }

            hasil_teks = status_map.get(angka_hasil, f"Kode: {angka_hasil}")

            return render_template('index.html', 
                                 prediction_text=f'Hasil Analisis: {hasil_teks}')
        
        except Exception as e:
            return render_template('index.html', 
                                 prediction_text=f'Kesalahan Sistem: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)