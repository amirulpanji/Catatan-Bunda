import os
import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Tentukan nama folder tempat menyimpan file pkl kamu
MODEL_DIR = 'models'

# LOAD SEMUA FILE MODEL, SCALER, DAN ENCODER
try:
    model_stunting = joblib.load(os.path.join(MODEL_DIR, 'model_knn_stunting.pkl'))
    scaler_stunting = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    
    # Pastikan file-file ini ada di dalam folder 'models' ya!
    model_personality = joblib.load(os.path.join(MODEL_DIR, 'model_personality.pkl'))
    scaler_personality = joblib.load(os.path.join(MODEL_DIR, 'scaler_personality.pkl'))
    le_drained = joblib.load(os.path.join(MODEL_DIR, 'le_drained.pkl'))
    le_stage = joblib.load(os.path.join(MODEL_DIR, 'le_stage.pkl'))
    le_personality = joblib.load(os.path.join(MODEL_DIR, 'le_personality.pkl'))
        
    print("Semua Model, Scaler, dan Encoder BERHASIL dimuat!")
except Exception as e:
    print(f"EROR KRITIKAL SAAT LOAD MODEL: {e}")


# ==============================================================================
# 2. ROUTES SETUP
# ==============================================================================
@app.route('/')
def home():
    return render_template('index.html')


# --- FITUR STUNTING (BAWAAN ORIGINAL) ---
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            umur = float(request.form['umur'])
            jk_input = request.form['jenis_kelamin']
            tinggi = float(request.form['tinggi'])

            if umur > 36:
                return render_template('index.html', 
                                       prediction_text="Maaf Bunda, umur maksimal adalah 36 bulan.",
                                       res_color="#ef476f")
            
            if tinggi < 45 or tinggi > 110:
                return render_template('index.html', 
                                       prediction_text="Tinggi harus antara 45cm - 110cm.",
                                       res_color="#ef476f")

            jk = 1 if jk_input == 'laki-laki' else 0
            fitur_raw = np.array([[umur, jk, tinggi]])
            fitur_scaled = scaler_stunting.transform(fitur_raw) # Menggunakan scaler khusus stunting
            
            prediksi = model_stunting.predict(fitur_scaled)
            angka_hasil = int(prediksi[0])

            status_config = {
                0: {"label": "Normal", "color": "#90ee90", "keterangan": "Pertumbuhan si kecil sesuai dengan grafik pertumbuhan standar.", "saran": "Pertahankan pola makan bergizi seimbang dan rutin kontrol ke Posyandu ya, Bun."},
                1: {"label": "Sangat Pendek", "color": "#ff0000", "keterangan": "Tinggi badan si kecil berada jauh di bawah rata-rata usianya.", "saran": "Segera konsultasikan dengan dokter spesialis anak atau Puskesmas untuk pemeriksaan lebih lanjut."},
                2: {"label": "Pendek", "color": "#ffb6c1", "keterangan": "Tinggi badan si kecil berada di bawah kurva pertumbuhan normal.", "saran": "Berikan tambahan protein hewani (telur, ikan, daging) dan pastikan kebersihan lingkungan terjaga."},
                3: {"label": "Tinggi", "color": "#00ff00", "keterangan": "Si kecil memiliki pertumbuhan tinggi badan yang sangat optimal.", "saran": "Luar biasa! Teruskan stimulasi motorik dan nutrisi sehatnya untuk mendukung kecerdasannya."}
            }

            res = status_config.get(angka_hasil, {"label": "Tidak Diketahui", "color": "#073b4c", "keterangan": "-", "saran": "-"})

            return render_template('index.html', 
                                   prediction_text=res['label'],
                                   res_color=res['color'],
                                   res_keterangan=res['keterangan'],
                                   res_saran=res['saran'])
        
        except ValueError:
            return render_template('index.html', prediction_text='Mohon masukkan angka yang valid, Bun.')
        except Exception as e:
            return render_template('index.html', prediction_text=f'Kesalahan Sistem: {str(e)}')
    return render_template('index.html')

# --- FITUR PERSONALITY (BARU) ---
@app.route('/predict-personality', methods=['POST'])
def predict_personality():
    if request.method == 'POST':
        try:
            # 1. Ambil Input Bertipe Angka/Slider dari HTML Form
            time_alone = float(request.form['time_alone'])
            social_events = float(request.form['social_events'])
            friends = float(request.form['friends'])
            going_out = float(request.form['going_out'])
            post_freq = float(request.form['post_freq'])
            
            # 2. Ambil Input Bertipe Teks (Dropdown Select)
            text_drained = request.form['drained']       # e.g., "Yes" atau "No"
            text_stage_fear = request.form['stage_fear'] # e.g., "High" atau "Low"

            # 3. Encoding Data Teks Menjadi Angka Menggunakan Label Encoder Colab
            drained_encoded = le_drained.transform([text_drained])[0]
            stage_fear_encoded = le_stage.transform([text_stage_fear])[0]

            # 4. Susun Array Sesuai Urutan 7 Kolom df Asli di Colab
            # SUSUN ULANG AGAR PAS DENGAN URUTAN COLAB KAMU:
            fitur_raw = np.array([[
                time_alone,          # 1. Time_spent_Alone
                stage_fear_encoded,  # 2. Stage_fear (Tadi ketukar ke paling bawah!)
                social_events,       # 3. Social_event_attendance
                going_out,           # 4. Going_outside
                drained_encoded,     # 5. Drained_after_socializing
                friends,             # 6. Friends_circle_size
                post_freq            # 7. Post_frequency
            ]])

            # 5. Scaling Menggunakan Scaler Khusus Personality
            fitur_scaled = scaler_personality.transform(fitur_raw)

            # 6. Prediksi Model KNN Personality
            prediksi_angka = model_personality.predict(fitur_scaled)

            # 7. Decode Balik Angka Prediksi Menjadi Teks ("Introvert" / "Extrovert")
            hasil_akhir = le_personality.inverse_transform(prediksi_angka)[0]

            # 8. Set Up Kostumisasi Tampilan Output Berdasarkan Hasil
            if hasil_akhir.lower() == 'introvert':
                warna_tampilan = "#4a90e2" # Biru tenang khas introvert
                keterangan_p = "Si Kecil kurang bergaul, cenderung memulihkan energi dengan menghabiskan waktu sendiri dan menyukai lingkaran pertemanan yang intim."
            else:
                warna_tampilan = "#f5a623" # Oranye ceria khas extrovert
                keterangan_p = "Si Kecil mendapatkan energi dari interaksi sosial, menyukai keramaian, dan sangat ekspresif terhadap lingkungan sekitar."

            # Render balik ke file HTML kamu (bisa diarahkan ke index.html atau template baru)
            return render_template('index.html', 
                                   pers_text=hasil_akhir,
                                   pers_color=warna_tampilan,
                                   pers_keterangan=keterangan_p)

        except ValueError:
            return render_template('index.html', pers_text='Mohon isi semua input kuesioner dengan benar, bro.')
        except Exception as e:
            return render_template('index.html', pers_text=f'Kesalahan Sistem Personality: {str(e)}')


if __name__ == "__main__":
    app.run(debug=True)