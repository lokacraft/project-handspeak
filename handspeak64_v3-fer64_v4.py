import os
import sys
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from gtts import gTTS
import pygame
import webbrowser
from collections import deque, Counter
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFrame, 
                             QTextEdit, QComboBox, QDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont

# --- KONFIGURASI PATH ---
HAND_MODEL_PATH = "models/handspeak_model_v3_11.keras"
FER_MODEL_PATH = "models/fer_emotion_model_v1.keras"
FEEDBACK_LINK = "https://docs.google.com/forms/d/e/1FAIpQLSc-DBQ1ExH3lijPvW5iGBzwwdWU30sV9EqV0qilwb1MZv70fQ/viewform?usp=dialog"
IMG_SIZE = 64

class VideoWorker(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    vision_signal = pyqtSignal(np.ndarray)
    detection_signal = pyqtSignal(str, float)
    emotion_signal = pyqtSignal(str, float)

    def __init__(self, hand_model, fer_model):
        super().__init__()
        self.hand_model = hand_model
        self.fer_model = fer_model
        self.running = True
        self.bg_mode = "None" 
        self.bg_color = (0, 0, 0)
        
        # Inisialisasi MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mp_face = mp.solutions.face_detection
        self.face_detector = self.mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)
        self.mp_selfie = mp.solutions.selfie_segmentation
        self.segment = self.mp_selfie.SelfieSegmentation(model_selection=1)
        
        self.alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self.emotion_labels = ['Bengong', 'Bingung', 'Positif', 'Pusing']
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def set_bg_mode(self, mode, color_bgr=(0,0,0)):
        self.bg_mode = mode
        self.bg_color = color_bgr

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
            
            # --- DETEKSI EMOSI ---
            face_results = self.face_detector.process(rgb_frame)
            current_emo, emo_c = "-", 0.0
            if face_results.detections:
                for d in face_results.detections:
                    b = d.location_data.relative_bounding_box
                    x, y, fw, fh = int(b.xmin*w), int(b.ymin*h), int(b.width*w), int(b.height*h)
                    x, y = max(0, x), max(0, y)
                    face_roi = frame[y:y+fh, x:x+fw]
                    if face_roi.size != 0 and self.fer_model:
                        try:
                            f_in = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
                            f_in = cv2.cvtColor(f_in, cv2.COLOR_BGR2RGB)
                            f_in = np.expand_dims(f_in, axis=0).astype('float32') / 255.0
                            preds = self.fer_model(f_in, training=False).numpy()
                            idx = np.argmax(preds)
                            current_emo, emo_c = self.emotion_labels[idx], preds[0][idx]
                            cv2.rectangle(frame, (x, y), (x+fw, y+fh), (0, 255, 0), 2)
                        except: pass
            
            self.emotion_signal.emit(current_emo, emo_c)

            # --- DETEKSI TANGAN ---
            hand_results = self.hands.process(rgb_frame)
            char, conf = "-", 0.0
            if hand_results.multi_hand_landmarks:
                for hl in hand_results.multi_hand_landmarks:
                    xl = [int(lm.x * w) for lm in hl.landmark]
                    yl = [int(lm.y * h) for lm in hl.landmark]
                    x1, x2 = max(0, min(xl)-50), min(w, max(xl)+50)
                    y1, y2 = max(0, min(yl)-50), min(h, max(yl)+50)
                    roi = frame[y1:y2, x1:x2]
                    if roi.size != 0:
                        if self.bg_mode != "None":
                            s_res = self.segment.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                            m = s_res.segmentation_mask > 0.3
                            bg = np.zeros(roi.shape, dtype=np.uint8); bg[:] = self.bg_color
                            roi = np.where(m[:, :, None], roi, bg)
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        ai_in = cv2.resize(self.clahe.apply(gray), (IMG_SIZE, IMG_SIZE))
                        self.vision_signal.emit(ai_in)
                        
                        if self.hand_model:
                            try:
                                h_in = ai_in.reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0
                                h_p = self.hand_model(h_in, training=False).numpy()
                                if np.max(h_p) > 0.85:
                                    char, conf = self.alphabet[np.argmax(h_p)], np.max(h_p)
                            except: pass

            self.detection_signal.emit(char, conf)
            self.change_pixmap_signal.emit(frame)
        cap.release()

class FeedbackModal(QDialog):
    def __init__(self, parent, emotion):
        super().__init__(parent)
        self.setWindowTitle("Sistem Bantuan")
        self.setFixedSize(400, 220)
        self.setStyleSheet("background: white; border-radius: 10px;")
        layout = QVBoxLayout(self)
        
        lbl = QLabel(f"💡 Kami mendeteksi Anda merasa <b>{emotion}</b>.<br>Apakah Anda membutuhkan bantuan atau ingin memberi masukan?", alignment=Qt.AlignCenter)
        lbl.setWordWrap(True); lbl.setFont(QFont('Segoe UI', 11))
        layout.addWidget(lbl)
        
        btn_form = QPushButton("Isi Form Feedback")
        btn_form.setStyleSheet("background: #007bff; color: white; height: 40px; font-weight: bold; border-radius: 5px;")
        btn_form.clicked.connect(lambda: [webbrowser.open(FEEDBACK_LINK), self.close()])
        layout.addWidget(btn_form)
        
        btn_close = QPushButton("Tutup")
        btn_close.clicked.connect(self.close)
        layout.addWidget(btn_close)

class HandSpeakApp(QMainWindow):
    def __init__(self):
        super().__init__()
        pygame.mixer.init()
        self.setWindowTitle("HandSpeak")
        self.setMinimumSize(1300, 850)
        self.setStyleSheet("background-color: #f8f9fa;")
        
        self.hand_model, self.fer_model, self.worker = None, None, None
        self.current_sentence, self.pending_char = "", ""
        
        # State Management
        self.emotion_counters = {"Bengong": 0, "Bingung": 0, "Pusing": 0}
        self.last_stable_emo = None
        self.modal_active = False
        
        self.init_models()
        self.init_ui()

    def init_models(self):
        try:
            if os.path.exists(HAND_MODEL_PATH):
                self.hand_model = tf.keras.models.load_model(HAND_MODEL_PATH, compile=False)
            if os.path.exists(FER_MODEL_PATH):
                self.fer_model = tf.keras.models.load_model(FER_MODEL_PATH, compile=False)
        except Exception as e: print(f"Error Models: {e}")

    def init_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        main_layout = QVBoxLayout(central); main_layout.setContentsMargins(30, 20, 30, 30)

        header = QLabel("HANDSPEAK AI Emotion Aware")
        header.setAlignment(Qt.AlignCenter); header.setFont(QFont('Segoe UI', 24, QFont.Bold))
        main_layout.addWidget(header)

        body = QHBoxLayout(); body.setSpacing(25)

        # LEFT PANEL
        left = QVBoxLayout()
        self.combo_bg = QComboBox(); self.combo_bg.addItems(["None (Real)", "Solid White", "Solid Black", "Solid Green"])
        left.addWidget(self.combo_bg)
        self.video_lbl = QLabel(); self.video_lbl.setFixedSize(640, 480); self.video_lbl.setStyleSheet("background: black; border-radius: 15px;")
        left.addWidget(self.video_lbl)
        self.btn_cam = QPushButton(" AKTIFKAN KAMERA"); self.btn_cam.setCheckable(True)
        self.btn_cam.setStyleSheet("background: #007bff; color: white; height: 50px; font-weight: bold; border-radius: 10px;")
        self.btn_cam.clicked.connect(self.toggle_camera); left.addWidget(self.btn_cam)
        body.addLayout(left)

        # RIGHT PANEL
        right = QVBoxLayout(); right.setSpacing(15)
        
        h_info = QHBoxLayout()
        # Membuat label secara manual agar aman dari AttributeError
        self.lbl_char_val = self.create_card(h_info, "HURUF", "#007bff")
        self.lbl_emo_val = self.create_card(h_info, "EMOSI", "#28a745")
        right.addLayout(h_info)

        # Vision Display
        self.lbl_vision = QLabel(); self.lbl_vision.setFixedSize(140, 140); self.lbl_vision.setStyleSheet("background: #212529; border-radius: 10px;")
        right.addWidget(QLabel("AI VISION (HAND)", alignment=Qt.AlignCenter))
        right.addWidget(self.lbl_vision, alignment=Qt.AlignCenter)

        # Result Box
        self.text_out = QTextEdit(); self.text_out.setFixedHeight(80); self.text_out.setFont(QFont('Segoe UI', 18)); self.text_out.setReadOnly(True)
        right.addWidget(self.text_out)
        
        btns = QHBoxLayout()
        btn_clr = QPushButton("CLEAR"); btn_clr.clicked.connect(self.clear_all); btns.addWidget(btn_clr)
        btn_speak = QPushButton("BACA"); btn_speak.setStyleSheet("background: #ffc107; font-weight: bold;")
        btn_speak.clicked.connect(self.speak); btns.addWidget(btn_speak)
        right.addLayout(btns)

        # Sign Guide - FIXED DISPLAY
        self.lbl_guide = QLabel(); self.lbl_guide.setAlignment(Qt.AlignCenter)
        if os.path.exists("assets/sign_guide.png"):
            pix = QPixmap("assets/sign_guide.png")
            # Menjamin gambar proporsional dan tidak terpotong
            self.lbl_guide.setPixmap(pix.scaled(550, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        right.addWidget(self.lbl_guide)

        body.addLayout(right); main_layout.addLayout(body)

    def create_card(self, layout, title, color):
        f = QFrame(); f.setStyleSheet(f"background: white; border-radius: 10px; border: 2px solid {color};")
        l = QVBoxLayout(f); l.addWidget(QLabel(title, alignment=Qt.AlignCenter, styleSheet="font-weight: bold;"))
        val = QLabel("-", alignment=Qt.AlignCenter); val.setFont(QFont('Segoe UI', 38, QFont.Bold)); val.setStyleSheet(f"color: {color};")
        l.addWidget(val); layout.addWidget(f)
        return val # Sekarang mengembalikan objek QLabel langsung

    def handle_emotion(self, emo, conf):
        # FIX: Akses label secara langsung, bukan lewat layout().itemAt()
        self.lbl_emo_val.setText(emo.upper())
        
        if emo in self.emotion_counters and not self.modal_active:
            if emo != self.last_stable_emo:
                self.emotion_counters[emo] += 1
                self.last_stable_emo = emo
                if self.emotion_counters[emo] >= 3:
                    self.show_feedback(emo)
        elif emo == "Positif":
            self.last_stable_emo = "Positif"

    def show_feedback(self, emo):
        self.modal_active = True
        self.emotion_counters[emo] = 0
        FeedbackModal(self, emo).exec_()
        self.modal_active = False

    def handle_detection(self, char, conf):
        # FIX: Akses label secara langsung
        self.lbl_char_val.setText(char)
        if char != "-" and char != "Y": self.pending_char = char
        if char == "Y" and self.pending_char != "":
            self.current_sentence += self.pending_char
            self.text_out.setText(self.current_sentence); self.pending_char = ""

    def toggle_camera(self):
        if self.btn_cam.isChecked():
            self.worker = VideoWorker(self.hand_model, self.fer_model)
            self.worker.change_pixmap_signal.connect(self.update_video)
            self.worker.vision_signal.connect(self.update_vision)
            self.worker.detection_signal.connect(self.handle_detection)
            self.worker.emotion_signal.connect(self.handle_emotion)
            self.worker.start()
        else:
            if self.worker: self.worker.stop(); self.video_lbl.clear()

    def speak(self):
        t = self.text_out.toPlainText()
        if t:
            try:
                if pygame.mixer.music.get_busy(): pygame.mixer.music.stop()
                pygame.mixer.music.unload()
                tts = gTTS(text=t, lang='id')
                if not os.path.exists("speech_output"): os.makedirs("speech_output")
                f = "speech_output/temp.mp3"; tts.save(f)
                pygame.mixer.music.load(f); pygame.mixer.music.play()
            except: pass

    def clear_all(self): self.current_sentence = ""; self.text_out.clear()
    def update_video(self, f): h, w, c = f.shape; q = QImage(f.data, w, h, c*w, QImage.Format_RGB888).rgbSwapped(); self.video_lbl.setPixmap(QPixmap.fromImage(q).scaled(640, 480, Qt.KeepAspectRatio))
    def update_vision(self, v): h, w = v.shape; q = QImage(v.data, w, h, w, QImage.Format_Grayscale8); self.lbl_vision.setPixmap(QPixmap.fromImage(q).scaled(140, 140, Qt.KeepAspectRatio))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HandSpeakApp(); window.show()
    sys.exit(app.exec_())