import os
import sys
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFrame, QGraphicsDropShadowEffect)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor

# Optimasi Lingkungan
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- 1. WORKER THREAD (DENGAN SIGNAL PREVIEW) ---
class VideoWorker(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    preview_signal = pyqtSignal(np.ndarray) # Signal baru untuk AI Vision
    detection_signal = pyqtSignal(str)

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.running = True
        self.mp_hands = mp.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

    def run(self):
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self.hands.process(rgb)
                
                char = "-"
                if res.multi_hand_landmarks:
                    for lms in res.multi_hand_landmarks:
                        mp.drawing_utils.draw_landmarks(frame, lms, self.mp_hands.HAND_CONNECTIONS)
                        
                        # ROI Extraction & Preprocessing
                        h, w, _ = frame.shape
                        x_list = [int(lm.x * w) for lm in lms.landmark]
                        y_list = [int(lm.y * h) for lm in lms.landmark]
                        xmin, xmax = max(0, min(x_list)-40), min(w, max(x_list)+40)
                        ymin, ymax = max(0, min(y_list)-40), min(h, max(y_list)+40)
                        
                        roi = frame[ymin:ymax, xmin:xmax]
                        if roi.size != 0:
                            # Proses ke Grayscale 28x28
                            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                            ai_input_view = cv2.resize(gray, (28, 28))
                            
                            # Kirim ke Preview Window
                            self.preview_signal.emit(ai_input_view)
                            
                            # Prediksi AI
                            input_data = (ai_input_view / 255.0).reshape(1, 28, 28, 1)
                            if self.model:
                                p = self.model.predict(input_data, verbose=0)
                                if np.max(p) > 0.85:
                                    char = "ABCDEFGHIKLMNOPQRSTUVWXY"[np.argmax(p)]
                
                self.detection_signal.emit(char)
                self.change_pixmap_signal.emit(frame)
        cap.release()

# --- 2. UI UTAMA ---
class HandSpeakApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HandSpeak AI - Thesis Debugger Mode")
        self.setGeometry(100, 100, 1300, 900)
        self.setStyleSheet("background-color: #fcfcfc;")
        self.model = None
        self.init_model()
        self.init_ui()

    def init_model(self):
        path = "models/handspeak_model_v3_11.keras"
        if os.path.exists(path):
            try:
                self.model = tf.keras.models.load_model(path, compile=False)
                print("Log: Model AI Loaded.")
            except Exception as e: print(f"Log ERROR: {e}")

    def init_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        layout = QVBoxLayout(central); layout.setContentsMargins(30, 20, 30, 30)

        # Header
        title = QLabel("HANDSPEAK AI TRANSLATOR"); title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont('Segoe UI', 26, QFont.Bold)); title.setStyleSheet("color: #2c3e50;")
        layout.addWidget(title)

        main_box = QHBoxLayout()
        
        # Left: Main Camera
        self.v_card = QFrame()
        self.v_card.setStyleSheet("background-color: white; border-radius: 15px; border: 1px solid #e0e0e0;")
        v_layout = QVBoxLayout(self.v_card)
        self.video_feed = QLabel(); self.video_feed.setFixedSize(640, 480)
        v_layout.addWidget(self.video_feed)
        main_box.addWidget(self.v_card)

        # Right Panel
        right_panel = QVBoxLayout(); right_panel.setSpacing(15)
        
        # Row: Detection & AI Vision
        row_top = QHBoxLayout()
        
        # Card 1: Detected Char
        self.char_card = QFrame()
        self.char_card.setStyleSheet("background-color: white; border-radius: 15px; border: 1px solid #e0e0e0;")
        c_layout = QVBoxLayout(self.char_card)
        c_layout.addWidget(QLabel("HURUF", alignment=Qt.AlignCenter))
        self.lbl_char = QLabel("-"); self.lbl_char.setFont(QFont('Segoe UI', 80, QFont.Bold))
        self.lbl_char.setStyleSheet("color: #3498db; border: none;"); self.lbl_char.setAlignment(Qt.AlignCenter)
        c_layout.addWidget(self.lbl_char)
        row_top.addWidget(self.char_card)

        # Card 2: AI Vision (What the AI sees)
        self.vision_card = QFrame()
        self.vision_card.setStyleSheet("background-color: #2c3e50; border-radius: 15px;")
        vis_layout = QVBoxLayout(self.vision_card)
        vis_layout.addWidget(QLabel("AI VISION (28x28)", alignment=Qt.AlignCenter, styleSheet="color: white; font-size: 10px;"))
        self.ai_vision_feed = QLabel(); self.ai_vision_feed.setFixedSize(120, 120) # Scaled up for eyes
        self.ai_vision_feed.setAlignment(Qt.AlignCenter)
        vis_layout.addWidget(self.ai_vision_feed)
        row_top.addWidget(self.vision_card)
        
        right_panel.addLayout(row_top)

        # Card 3: Sign Guide
        self.guide_card = QFrame()
        self.guide_card.setStyleSheet("background-color: white; border-radius: 15px; border: 1px solid #e0e0e0;")
        g_layout = QVBoxLayout(self.guide_card)
        self.guide_img = QLabel()
        g_path = "assets/sign_guide.png"
        if os.path.exists(g_path):
            self.guide_img.setPixmap(QPixmap(g_path).scaled(380, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        g_layout.addWidget(self.guide_img)
        right_panel.addWidget(self.guide_card)

        self.btn_start = QPushButton("AKTIFKAN KAMERA")
        self.btn_start.setStyleSheet("background-color: #3498db; color: white; font-weight: bold; height: 55px; border-radius: 10px;")
        self.btn_start.clicked.connect(self.start_ai)
        right_panel.addWidget(self.btn_start)

        main_box.addLayout(right_panel)
        layout.addLayout(main_box)

    def start_ai(self):
        self.worker = VideoWorker(self.model)
        self.worker.change_pixmap_signal.connect(self.update_main_video)
        self.worker.preview_signal.connect(self.update_ai_vision)
        self.worker.detection_signal.connect(self.lbl_char.setText)
        self.worker.start()
        self.btn_start.setEnabled(False); self.btn_start.setText("SISTEM AKTIF")

    def update_main_video(self, frame):
        h, w, ch = frame.shape
        img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888).rgbSwapped()
        self.video_feed.setPixmap(QPixmap.fromImage(img).scaled(640, 480, Qt.KeepAspectRatio))

    def update_ai_vision(self, frame_28x28):
        # Mengubah grayscale 28x28 menjadi QImage agar bisa tampil
        h, w = frame_28x28.shape
        img = QImage(frame_28x28.data, w, h, w, QImage.Format_Grayscale8)
        self.ai_vision_feed.setPixmap(QPixmap.fromImage(img).scaled(120, 120, Qt.KeepAspectRatio))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HandSpeakApp(); window.show()
    sys.exit(app.exec_())