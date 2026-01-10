import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFrame, 
                             QTextEdit, QGraphicsDropShadowEffect, QStyle)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor

# Optimasi Lingkungan
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- 1. WORKER THREAD (PURE COMPUTER VISION - NO MEDIAPIPE) ---
class VideoWorker(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    vision_signal = pyqtSignal(np.ndarray)
    detection_signal = pyqtSignal(str)

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.running = True
        self.alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self.roi_size = 320 # Ukuran kotak deteksi di layar

    def run(self):
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                
                # Tentukan koordinat Kotak Statis (Tengah Layar)
                x1 = (w // 2) - (self.roi_size // 2)
                y1 = (h // 2) - (self.roi_size // 2)
                x2, y2 = x1 + self.roi_size, y1 + self.roi_size
                
                # 1. Ekstraksi ROI (Region of Interest)
                roi = frame[y1:y2, x1:x2]
                
                char = "-"
                if roi.size != 0:
                    # 2. Preprocessing Murni (Grayscale & 64x64)
                    # Sesuai permintaan: Biarkan model memproses grayscale
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    
                    # Normalisasi Cahaya agar mirip dataset Colab
                    gray = cv2.equalizeHist(gray) 
                    
                    # Resize ke 64x64
                    ai_input_view = cv2.resize(gray, (64, 64))
                    
                    # Kirim sinyal ke AI Vision Preview (untuk debugging)
                    self.vision_signal.emit(ai_input_view)
                    
                    # 3. Prediksi AI
                    # Reshape ke (1, 64, 64, 1) dan normalisasi 0-1
                    final_input = ai_input_view.reshape(1, 64, 64, 1) / 255.0
                    
                    if self.model:
                        pred = self.model.predict(final_input, verbose=0)
                        confidence = np.max(pred)
                        if confidence > 0.85: # Threshold akurasi
                            char = self.alphabet[np.argmax(pred)]
                
                # Gambar visualisasi kotak di frame kamera
                cv2.rectangle(frame, (x1, y1), (x2, y2), (46, 204, 113), 3)
                cv2.putText(frame, "POSISI TANGAN DI SINI", (x1, y1-15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (46, 204, 113), 2)

                self.detection_signal.emit(char)
                self.change_pixmap_signal.emit(frame)
        cap.release()

# --- 2. UI UTAMA (MODERN LIGHT THEME) ---
class HandSpeakApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HandSpeak AI - Stable Vision Mode")
        self.setMinimumSize(1300, 850)
        self.setStyleSheet("background-color: #fcfcfc;")
        
        self.model = None
        self.init_model()
        self.init_ui()

    def init_model(self):
        # Memuat model .keras versi terbaru
        path = "models/handspeak_model_v3_11.keras"
        if os.path.exists(path):
            try:
                self.model = tf.keras.models.load_model(path, compile=False)
                print("Log: Model AI 64x64 Berhasil Dimuat.")
            except Exception as e: print(f"Log ERROR: {e}")

    def init_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        layout = QVBoxLayout(central); layout.setContentsMargins(40, 20, 40, 40)

        # Header
        header = QLabel("HANDSPEAK AI TRANSLATOR"); header.setAlignment(Qt.AlignCenter)
        header.setFont(QFont('Segoe UI', 28, QFont.Bold)); header.setStyleSheet("color: #2d3436;")
        layout.addWidget(header)

        body = QHBoxLayout(); body.setSpacing(30)

        # --- PANEL KIRI: VIDEO ---
        left_side = QVBoxLayout()
        self.video_card = QFrame()
        self.video_card.setStyleSheet("background-color: white; border-radius: 20px; border: 1px solid #dfe6e9;")
        v_lay = QVBoxLayout(self.video_card)
        self.lbl_video = QLabel(); self.lbl_video.setFixedSize(640, 480)
        self.lbl_video.setStyleSheet("background-color: #000; border-radius: 10px;")
        v_lay.addWidget(self.lbl_video)
        left_side.addWidget(self.video_card)

        self.btn_start = QPushButton(" AKTIFKAN KAMERA")
        self.btn_start.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.btn_start.setStyleSheet("""
            QPushButton {
                background-color: #0984e3; color: white; font-weight: bold; font-size: 18px;
                height: 65px; border-radius: 15px; border: none;
            }
            QPushButton:hover { background-color: #74b9ff; }
        """)
        self.btn_start.clicked.connect(self.start_system)
        left_side.addWidget(self.btn_start)
        body.addLayout(left_side)

        # --- PANEL KANAN: TOOLS ---
        right_side = QVBoxLayout(); right_side.setSpacing(20)

        # Row 1: Char & Vision
        info_row = QHBoxLayout()
        
        # Detected Char
        self.char_card = QFrame()
        self.char_card.setStyleSheet("background-color: white; border-radius: 20px; border: 1px solid #dfe6e9;")
        c_lay = QVBoxLayout(self.char_card)
        c_lay.addWidget(QLabel("HURUF", alignment=Qt.AlignCenter, styleSheet="color: #636e72; font-weight: bold;"))
        self.lbl_char = QLabel("-"); self.lbl_char.setFont(QFont('Segoe UI', 100, QFont.Bold))
        self.lbl_char.setStyleSheet("color: #0984e3; border: none;"); self.lbl_char.setAlignment(Qt.AlignCenter)
        c_lay.addWidget(self.lbl_char)
        info_row.addWidget(self.char_card)

        # AI Vision 64x64
        self.vis_card = QFrame()
        self.vis_card.setStyleSheet("background-color: #2d3436; border-radius: 20px;")
        self.vis_card.setFixedSize(180, 220)
        vis_lay = QVBoxLayout(self.vis_card)
        vis_lay.addWidget(QLabel("AI VISION", alignment=Qt.AlignCenter, styleSheet="color: #dfe6e9; font-size: 10px;"))
        self.lbl_vision = QLabel(); self.lbl_vision.setFixedSize(128, 128)
        vis_lay.addWidget(self.lbl_vision, alignment=Qt.AlignCenter)
        info_row.addWidget(self.vis_card)
        right_side.addLayout(info_row)

        # Row 2: Sentence Output
        self.res_card = QFrame()
        self.res_card.setStyleSheet("background-color: white; border-radius: 20px; border: 1px solid #dfe6e9;")
        res_lay = QVBoxLayout(self.res_card)
        res_lay.addWidget(QLabel("HASIL KONFIRMASI", styleSheet="color: #636e72; font-weight: bold;"))
        self.text_out = QTextEdit(); self.text_out.setFixedHeight(80)
        self.text_out.setFont(QFont('Segoe UI', 22)); self.text_out.setStyleSheet("border: none; color: #2d3436;")
        res_lay.addWidget(self.text_out)
        right_side.addWidget(self.res_card)

        # Row 3: Guide Image
        self.guide_card = QFrame()
        self.guide_card.setStyleSheet("background-color: white; border-radius: 20px; border: 1px solid #dfe6e9;")
        g_lay = QVBoxLayout(self.guide_card)
        self.lbl_guide = QLabel()
        g_path = "assets/sign_guide.png"
        if os.path.exists(g_path):
            self.lbl_guide.setPixmap(QPixmap(g_path).scaled(420, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        g_lay.addWidget(self.lbl_guide, alignment=Qt.AlignCenter)
        right_side.addWidget(self.guide_card)

        body.addLayout(right_side)
        layout.addLayout(body)

        # Efek Shadow
        for w in [self.video_card, self.char_card, self.res_card, self.guide_card]:
            shadow = QGraphicsDropShadowEffect()
            shadow.setBlurRadius(20); shadow.setColor(QColor(0,0,0,30))
            w.setGraphicsEffect(shadow)

    def start_system(self):
        self.worker = VideoWorker(self.model)
        self.worker.change_pixmap_signal.connect(self.update_main_video)
        self.worker.vision_signal.connect(self.update_ai_vision)
        self.worker.detection_signal.connect(self.lbl_char.setText)
        self.worker.start()
        self.btn_start.setEnabled(False); self.btn_start.setText(" SISTEM AKTIF")
        self.btn_start.setStyleSheet("background-color: #00b894; color: white; font-weight: bold; height: 65px; border-radius: 15px;")

    def update_main_video(self, frame):
        h, w, ch = frame.shape
        img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888).rgbSwapped()
        self.lbl_video.setPixmap(QPixmap.fromImage(img).scaled(640, 480, Qt.KeepAspectRatio))

    def update_ai_vision(self, frame_64):
        h, w = frame_64.shape
        img = QImage(frame_64.data, w, h, w, QImage.Format_Grayscale8)
        self.lbl_vision.setPixmap(QPixmap.fromImage(img).scaled(128, 128, Qt.KeepAspectRatio))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HandSpeakApp(); window.show()
    sys.exit(app.exec_())