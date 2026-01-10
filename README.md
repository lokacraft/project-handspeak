# HandSpeak AI - Sign Language Translator

Proyek ini adalah sistem penterjemah bahasa isyarat menggunakan algoritma **Convolutional Neural Network (CNN)** dengan integrasi **MediaPipe** untuk pelacakan tangan dinamis.

## Fitur Utama
- **Dynamic ROI Tracking**: Kotak deteksi mengikuti gerakan tangan secara otomatis.
- **Intelligent Preprocessing**: Menggunakan filter CLAHE untuk reduksi noise latar belakang.
- **Temporal Voting**: Menstabilkan hasil prediksi agar tidak melompat-lompat.
- **Solid Background Masking**: Pilihan latar belakang solid untuk meningkatkan kontras.
- **Text-to-Speech (TTS)**: Mampu menyuarakan hasil konfirmasi kalimat.

## Prasyarat
- **Python 3.11** (Sangat disarankan)
- Kamera Web (Internal/Eksternal)
- Koneksi Internet (Hanya untuk fitur Suara/TTS)

## Panduan Konfigurasi Sistem

### 1. Persiapan Lingkungan Virtual (Venv)
Gunakan Virtual Environment agar library proyek ini tidak bentrok dengan sistem global Anda.

```bash
# Membuat environment baru
python -m venv env_handspeak

# Aktivasi Environment (Windows)
env_handspeak\Scripts\activate

# Aktivasi Environment (Linux/macOS/Git Bash)
source env_handspeak/Scripts/activate

### 2. Install Library Lengkap
python -m pip install --upgrade pip
pip install -r requirements.txt

### 3. Struktur folder
project-handspeak/
├── env_handspeak/          # Folder Virtual Environment
├── models/                 
│   └── handspeak_model_v3_11.keras  # Otak AI (CNN 64x64)
├── assets/                 
│   └── sign_guide.png      # Gambar panduan visual isyarat SIBI
├── handspeak64_v2.py       # Aplikasi Versi Stabil (Basic)
├── handspeak64_v3.py       # Aplikasi Versi Ultimate (Background Masking + TTS)
├── requirements.txt        # Daftar library lengkap
└── README.md               # Dokumentasi proyek