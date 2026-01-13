import os
import sys
import cv2
import numpy as np
import mediapipe as mp

# Menggunakan tf.compat.v1 untuk menghilangkan peringatan reset_default_graph
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from collections import deque, Counter
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFrame, 
                             QTextEdit, QComboBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont
import pygame
from gtts import gTTS

# --- KONFIGURASI PATH ---
HAND_MODEL_PATH = "models/handspeak_model_v3_11.keras"
FER_MODEL_PATH = "models/fer_emotion_model_v1.keras"
IMG_SIZE = 64

class VideoWorker(QThread):
    # Signal menggunakan 'object' agar kompatibel dengan Python 3.11
    change_pixmap_signal = pyqtSignal(object)
    vision_signal = pyqtSignal(object)
    detection_signal = pyqtSignal(str, float)
    emotion_signal = pyqtSignal(str, float)

    def __init__(self, hand_model, fer_model):
        super().__init__()
        self.hand_model = hand_model
        self.fer_model = fer_model
        self.running = True
        
        # Inisialisasi MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True)
        
        self.alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self.emotion_labels = ['Bengong', 'Bingung', 'Positif', 'Pusing']
        self.prediction_buffer = deque(maxlen=10)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def stop(self):
        self.running = False
        self.wait()

    def run(self):
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 1. PROSES EMOSI (KOTAK HIJAU)
            face_results = self.face_mesh.process(rgb_frame)
            if face_results.multi_face_landmarks and self.fer_model:
                for face_lms in face_results.multi_face_landmarks:
                    x_c = [int(lm.x * w) for lm in face_lms.landmark]
                    y_c = [int(lm.y * h) for lm in face_lms.landmark]
                    x_min, x_max = max(0, min(x_c)-20), min(w, max(x_c)+20)
                    y_min, y_max = max(0, min(y_c)-20), min(h, max(y_c)+20)
                    
                    face_roi = frame[y_min:y_max, x_min:x_max]
                    if face_roi.size != 0:
                        f_in = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
                        f_in = f_in.reshape(1, IMG_SIZE, IMG_SIZE, 3) / 255.0
                        
                        # SOLUSI ERROR dense_2: Menggunakan __call__ secara eksplisit
                        preds = self.fer_model(f_in, training=False).numpy()
                        self.emotion_signal.emit(self.emotion_labels[np.argmax(preds)], float(np.max(preds)))
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # 2. PROSES TANGAN (KOTAK BIRU)
            hand_results = self.hands.process(rgb_frame)
            char, conf = "-", 0.0
            if hand_results.multi_hand_landmarks:
                for hand_lms in hand_results.multi_hand_landmarks:
                    x_l = [int(lm.x * w) for lm in hand_lms.landmark]
                    y_l = [int(lm.y * h) for lm in hand_lms.landmark]
                    x_mi, x_ma = max(0, min(x_l)-50), min(w, max(x_l)+50)
                    y_mi, y_ma = max(0, min(y_l)-50), min(h, max(y_l)+50)
                    
                    roi = frame[y_mi:y_ma, x_mi:x_ma]
                    if roi.size != 0:
                        cv2.rectangle(frame, (x_mi, y_mi), (x_ma, y_ma), (255, 0, 0), 2)
                        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        ai_in = cv2.resize(self.clahe.apply(gray), (IMG_SIZE, IMG_SIZE))
                        self.vision_signal.emit(ai_in)
                        
                        if self.hand_model:
                            h_in = ai_in.reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0
                            h_preds = self.hand_model(h_in, training=False).numpy()
                            if np.max(h_preds) > 0.85:
                                char, conf = self.alphabet[np.argmax(h_preds)], np.max(h_preds)
                                self.prediction_buffer.append(char)

            if self.prediction_buffer:
                char = Counter(self.prediction_buffer).most_common(1)[0][0]

            self.detection_signal.emit(char, conf)
            self.change_pixmap_signal.emit(frame)
        cap.release()

class HandSpeakApp(QMainWindow):
    def __init__(self):
        super().__init__()
        pygame.mixer.init()
        self.setWindowTitle("HandSpeak AI Multimodal - Unjani IR Project")
        self.setMinimumSize(1300, 850)
        self.setStyleSheet("background-color: #f4f7f6;")
        self.hand_model, self.fer_model, self.worker = None, None, None
        self.current_sentence, self.pending_char = "", ""
        
        self.init_models()
        self.init_ui()

    def init_models(self):
        try:
            # 1. Menghilangkan peringatan deprecated reset_default_graph
            tf.compat.v1.reset_default_graph() 
            K.clear_session()
            
            # 2. Muat model dengan compile=False agar tidak bentrok optimizer
            if os.path.exists(HAND_MODEL_PATH):
                self.hand_model = keras.models.load_model(HAND_MODEL_PATH, compile=False)
            if os.path.exists(FER_MODEL_PATH):
                self.fer_model = keras.models.load_model(FER_MODEL_PATH, compile=False)
            print("Log: Sistem AI Berhasil Dimuat Tanpa Konflik Layer.")
        except Exception as e: print(f"Error Model: {e}")

    def init_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        main_layout = QVBoxLayout(central); main_layout.setContentsMargins(30, 20, 30, 30)
        
        header = QLabel("HANDSPEAK AI + FER MULTIMODAL"); header.setAlignment(Qt.AlignCenter)
        header.setFont(QFont('Segoe UI', 22, QFont.Bold)); main_layout.addWidget(header)

        body = QHBoxLayout()
        # Left Panel (Video)
        left = QVBoxLayout()
        self.video_lbl = QLabel(); self.video_lbl.setFixedSize(640, 480); self.video_lbl.setStyleSheet("background: #000; border-radius: 15px;")
        left.addWidget(self.video_lbl)
        self.btn_cam = QPushButton(" AKTIFKAN KAMERA"); self.btn_cam.setCheckable(True); self.btn_cam.setFixedHeight(55)
        self.btn_cam.setStyleSheet("background: #007bff; color: white; border-radius: 10px; font-weight: bold;")
        self.btn_cam.clicked.connect(self.toggle_camera); left.addWidget(self.btn_cam)
        body.addLayout(left)

        # Right Panel (Analysis & Buttons)
        right = QVBoxLayout(); right.setSpacing(15)
        h_cards = QHBoxLayout()
        self.char_lbl = self.create_card(h_cards, "HURUF", "blue")
        self.emo_lbl = self.create_card(h_cards, "EMOSI", "green")
        right.addLayout(h_cards)

        self.text_out = QTextEdit(); self.text_out.setFixedHeight(120); self.text_out.setFont(QFont('Segoe UI', 18))
        right.addWidget(QLabel("HASIL KALIMAT:", styleSheet="font-weight: bold;"))
        right.addWidget(self.text_out)

        # FITUR TOMBOL LENGKAP
        btns = QHBoxLayout()
        btn_speak = QPushButton(" BACA"); btn_speak.clicked.connect(self.speak); btn_speak.setStyleSheet("background: #ffc107; height: 50px; font-weight: bold;")
        btn_del = QPushButton(" HAPUS"); btn_del.clicked.connect(self.delete_char); btn_del.setStyleSheet("background: #f8f9fa; height: 50px; border: 1px solid #dee2e6;")
        btn_clr = QPushButton(" CLEAR ALL"); btn_clr.clicked.connect(self.clear); btn_clr.setStyleSheet("background: #dc3545; color: white; height: 50px;")
        btns.addWidget(btn_speak); btns.addWidget(btn_del); btns.addWidget(btn_clr)
        right.addLayout(btns)
        
        body.addLayout(right); main_layout.addLayout(body)

    def create_card(self, layout, title, color):
        card = QFrame(); card.setStyleSheet(f"background: white; border-radius: 10px; border: 2px solid {color};")
        l = QVBoxLayout(card); l.addWidget(QLabel(title, alignment=Qt.AlignCenter, styleSheet="font-weight: bold;"))
        val = QLabel("-", alignment=Qt.AlignCenter); val.setFont(QFont('Segoe UI', 30, QFont.Bold))
        l.addWidget(val); layout.addWidget(card)
        return val

    def toggle_camera(self):
        if self.btn_cam.isChecked():
            self.worker = VideoWorker(self.hand_model, self.fer_model)
            self.worker.change_pixmap_signal.connect(self.update_video)
            self.worker.detection_signal.connect(self.handle_detection)
            self.worker.emotion_signal.connect(self.update_emotion)
            self.worker.start()
        else:
            if self.worker: self.worker.stop()

    def update_video(self, f):
        h, w, ch = f.shape; q = QImage(f.data, w, h, ch*w, QImage.Format_RGB888).rgbSwapped()
        self.video_lbl.setPixmap(QPixmap.fromImage(q).scaled(640, 480, Qt.KeepAspectRatio))

    def handle_detection(self, char, conf):
        self.char_lbl.setText(char)
        if char != "-" and char != "Y": self.pending_char = char
        if char == "Y" and self.pending_char:
            self.current_sentence += self.pending_char
            self.text_out.setText(self.current_sentence); self.pending_char = ""

    def update_emotion(self, emo, conf):
        self.emo_lbl.setText(emo.upper())

    def speak(self):
        txt = self.text_out.toPlainText()
        if txt:
            if not os.path.exists("speech_output"): os.makedirs("speech_output")
            gTTS(text=txt, lang='id').save("speech_output/temp.mp3")
            pygame.mixer.music.load("speech_output/temp.mp3"); pygame.mixer.music.play()

    def delete_char(self): 
        self.current_sentence = self.current_sentence[:-1]
        self.text_out.setText(self.current_sentence)

    def clear(self): 
        self.current_sentence = ""
        self.text_out.clear()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HandSpeakApp(); window.show()
    sys.exit(app.exec_())