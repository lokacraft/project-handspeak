import os
import sys
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque, Counter
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFrame, 
                             QTextEdit, QGraphicsDropShadowEffect, QStyle)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor

# --- KONFIGURASI ---
MODEL_PATH = "models/handspeak_model_v3_11.keras"
IMG_SIZE = 64
BUFFER_SIZE = 10 

class VideoWorker(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    vision_signal = pyqtSignal(np.ndarray)
    detection_signal = pyqtSignal(str, float)

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.running = True
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, max_num_hands=1,
            min_detection_confidence=0.7, min_tracking_confidence=0.7
        )
        self.alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self.prediction_buffer = deque(maxlen=BUFFER_SIZE)
        self.confidence_buffer = deque(maxlen=BUFFER_SIZE)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def stop(self):
        self.running = False
        self.wait()

    def run(self):
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                
                final_char = "-"
                final_conf = 0.0
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        x_list = [int(lm.x * w) for lm in hand_landmarks.landmark]
                        y_list = [int(lm.y * h) for lm in hand_landmarks.landmark]
                        x_min, x_max = max(0, min(x_list)-50), min(w, max(x_list)+50)
                        y_min, y_max = max(0, min(y_list)-50), min(h, max(y_list)+50)
                        
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        
                        roi = frame[y_min:y_max, x_min:x_max]
                        if roi.size != 0:
                            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                            enhanced = self.clahe.apply(gray)
                            ai_input = cv2.resize(enhanced, (IMG_SIZE, IMG_SIZE))
                            self.vision_signal.emit(ai_input)
                            
                            inp = ai_input.reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0
                            if self.model:
                                preds = self.model.predict(inp, verbose=0)
                                conf = np.max(preds)
                                if conf > 0.80:
                                    char = self.alphabet[np.argmax(preds)]
                                    self.prediction_buffer.append(char)
                                    self.confidence_buffer.append(conf)
                
                if self.prediction_buffer:
                    most_common = Counter(self.prediction_buffer).most_common(1)
                    final_char = most_common[0][0]
                    final_conf = np.mean(self.confidence_buffer)

                self.detection_signal.emit(final_char, final_conf)
                self.change_pixmap_signal.emit(frame)
        cap.release()

class HandSpeakApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HandSpeak AI - Thesis Edition")
        self.setMinimumSize(1350, 900)
        self.setStyleSheet("background-color: #f8f9fa;")
        
        self.model = None
        self.worker = None
        self.current_sentence = ""
        self.pending_char = ""
        
        self.init_model()
        self.init_ui()

    def init_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                # Memperbaiki deserialization batch_shape
                self.model = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
                print("Log: Model loaded successfully.")
            except Exception as e: print(f"Error Model: {e}")

    def init_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        main_layout = QVBoxLayout(central); main_layout.setContentsMargins(30, 20, 30, 30)

        header = QLabel("HANDSPEAK AI TRANSLATOR"); header.setAlignment(Qt.AlignCenter)
        header.setFont(QFont('Segoe UI', 26, QFont.Bold)); header.setStyleSheet("color: #2c3e50;")
        main_layout.addWidget(header)

        body = QHBoxLayout(); body.setSpacing(25)

        # --- LEFT PANEL (Kamera) ---
        left_panel = QVBoxLayout()
        self.v_card = QFrame(); self.v_card.setStyleSheet("background: white; border-radius: 20px; border: 1px solid #dee2e6;")
        v_lay = QVBoxLayout(self.v_card)
        self.video_lbl = QLabel(); self.video_lbl.setFixedSize(640, 480); self.video_lbl.setStyleSheet("background: #000; border-radius: 10px;")
        v_lay.addWidget(self.video_lbl)
        left_panel.addWidget(self.v_card)

        self.btn_cam = QPushButton(" AKTIFKAN KAMERA")
        self.btn_cam.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.btn_cam.setCheckable(True)
        self.btn_cam.setStyleSheet("""
            QPushButton { background: #007bff; color: white; height: 60px; border-radius: 15px; font-weight: bold; font-size: 16px; }
            QPushButton:checked { background: #dc3545; }
        """)
        self.btn_cam.clicked.connect(self.toggle_camera)
        left_panel.addWidget(self.btn_cam)
        body.addLayout(left_panel)

        # --- RIGHT PANEL (Analytics) ---
        right_panel = QVBoxLayout(); right_panel.setSpacing(20)

        info_h = QHBoxLayout()
        # Card Karakter Terdeteksi
        self.char_card = QFrame(); self.char_card.setStyleSheet("background: white; border-radius: 20px; border: 1px solid #dee2e6;")
        c_lay = QVBoxLayout(self.char_card)
        c_lay.addWidget(QLabel("HURUF TERDETEKSI", alignment=Qt.AlignCenter, styleSheet="color: #6c757d; font-weight: bold;"))
        self.lbl_char = QLabel("-"); self.lbl_char.setFont(QFont('Segoe UI', 80, QFont.Bold)); self.lbl_char.setStyleSheet("color: #007bff;")
        c_lay.addWidget(self.lbl_char, alignment=Qt.AlignCenter)
        self.lbl_conf = QLabel("Akurasi: 0%"); self.lbl_conf.setStyleSheet("color: #28a745; font-weight: bold;")
        c_lay.addWidget(self.lbl_conf, alignment=Qt.AlignCenter)
        info_h.addWidget(self.char_card)

        # AI Vision Preview
        self.vis_card = QFrame(); self.vis_card.setStyleSheet("background: #212529; border-radius: 20px;")
        self.vis_card.setFixedSize(180, 230)
        vi_lay = QVBoxLayout(self.vis_card)
        vi_lay.addWidget(QLabel("AI VISION", alignment=Qt.AlignCenter, styleSheet="color: #adb5bd; font-size: 10px;"))
        self.lbl_vision = QLabel(); self.lbl_vision.setFixedSize(140, 140)
        vi_lay.addWidget(self.lbl_vision, alignment=Qt.AlignCenter)
        info_h.addWidget(self.vis_card)
        right_panel.addLayout(info_h)

        # Sentence Builder (HALO Mode)
        self.res_card = QFrame(); self.res_card.setStyleSheet("background: white; border-radius: 20px; border: 1px solid #dee2e6;")
        res_lay = QVBoxLayout(self.res_card)
        res_lay.addWidget(QLabel("KONFIRMASI HASIL ('Y' UNTUK NEXT)", styleSheet="font-weight: bold; color: #6c757d;"))
        self.text_out = QTextEdit(); self.text_out.setFixedHeight(90); self.text_out.setFont(QFont('Segoe UI', 24))
        self.text_out.setReadOnly(True); self.text_out.setStyleSheet("border: none; color: #343a40;")
        res_lay.addWidget(self.text_out)
        
        btn_h = QHBoxLayout()
        # FIX: Menggunakan SP_DialogDiscardButton untuk mengganti SP_Trash
        self.btn_del = QPushButton(" HAPUS"); self.btn_del.setIcon(self.style().standardIcon(QStyle.SP_DialogDiscardButton))
        self.btn_del.clicked.connect(self.delete_last_char)
        self.btn_clear = QPushButton(" BERSIHKAN"); self.btn_clear.setIcon(self.style().standardIcon(QStyle.SP_DialogResetButton))
        self.btn_clear.clicked.connect(self.clear_all)
        
        for b in [self.btn_del, self.btn_clear]:
            b.setStyleSheet("background: #f8f9fa; border: 1px solid #dee2e6; height: 45px; border-radius: 12px; font-weight: bold;")
            btn_h.addWidget(b)
        res_lay.addLayout(btn_h)
        right_panel.addWidget(self.res_card)

        # Card: Guide Image
        self.guide_card = QFrame(); self.guide_card.setStyleSheet("background: white; border-radius: 20px; border: 1px solid #dee2e6;")
        g_lay = QVBoxLayout(self.guide_card)
        self.lbl_guide = QLabel()
        if os.path.exists("assets/sign_guide.png"):
            self.lbl_guide.setPixmap(QPixmap("assets/sign_guide.png").scaled(400, 200, Qt.KeepAspectRatio))
        g_lay.addWidget(self.lbl_guide, alignment=Qt.AlignCenter)
        right_panel.addWidget(self.guide_card)

        body.addLayout(right_panel)
        main_layout.addLayout(body)

        # Tambahkan Shadow Effect
        for w in [self.v_card, self.char_card, self.res_card, self.guide_card]:
            s = QGraphicsDropShadowEffect(); s.setBlurRadius(15); s.setColor(QColor(0,0,0,30)); w.setGraphicsEffect(s)

    def toggle_camera(self):
        if self.btn_cam.isChecked():
            self.btn_cam.setText(" MATIKAN KAMERA")
            self.worker = VideoWorker(self.model)
            self.worker.change_pixmap_signal.connect(self.update_video_frame)
            self.worker.vision_signal.connect(self.update_vision_frame)
            self.worker.detection_signal.connect(self.handle_logic)
            self.worker.start()
        else:
            self.btn_cam.setText(" AKTIFKAN KAMERA")
            if self.worker: self.worker.stop()
            self.video_lbl.clear()

    def handle_logic(self, char, conf):
        self.lbl_char.setText(char)
        self.lbl_conf.setText(f"Akurasi: {conf*100:.1f}%")
        
        # Logika 'HyAyLyO'
        if char != "-" and char != "Y":
            self.pending_char = char
        
        if char == "Y" and self.pending_char != "":
            self.current_sentence += self.pending_char
            self.text_out.setText(self.current_sentence)
            self.pending_char = ""

    def delete_last_char(self):
        self.current_sentence = self.current_sentence[:-1]
        self.text_out.setText(self.current_sentence)

    def clear_all(self):
        self.current_sentence = ""
        self.text_out.clear()

    def update_video_frame(self, frame):
        h, w, ch = frame.shape
        img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888).rgbSwapped()
        self.video_lbl.setPixmap(QPixmap.fromImage(img).scaled(640, 480, Qt.KeepAspectRatio))

    def update_vision_frame(self, f64):
        h, w = f64.shape
        img = QImage(f64.data, w, h, w, QImage.Format_Grayscale8)
        self.lbl_vision.setPixmap(QPixmap.fromImage(img).scaled(140, 140, Qt.KeepAspectRatio))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HandSpeakApp(); window.show()
    sys.exit(app.exec_())