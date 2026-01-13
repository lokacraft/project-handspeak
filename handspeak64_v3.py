import os
import sys
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from gtts import gTTS
import pygame
from collections import deque, Counter
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFrame, 
                             QTextEdit, QGraphicsDropShadowEffect, QStyle, QComboBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor

#  KONFIGURASI 
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
        self.bg_mode = "None" 
        self.bg_color = (0, 0, 0)
        
        # Inisialisasi MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, max_num_hands=1,
            min_detection_confidence=0.7, min_tracking_confidence=0.7
        )
        self.mp_selfie = mp.solutions.selfie_segmentation
        self.segment = self.mp_selfie.SelfieSegmentation(model_selection=1)
        
        self.alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self.prediction_buffer = deque(maxlen=BUFFER_SIZE)
        self.confidence_buffer = deque(maxlen=BUFFER_SIZE)
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
                        
                        roi = frame[y_min:y_max, x_min:x_max]
                        if roi.size != 0:
                            #  Background Masking
                            if self.bg_mode != "None":
                                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                                seg_results = self.segment.process(roi_rgb)
                                mask = seg_results.segmentation_mask > 0.3
                                bg_image = np.zeros(roi.shape, dtype=np.uint8)
                                bg_image[:] = self.bg_color
                                roi_final = np.where(mask[:, :, None], roi, bg_image)
                            else:
                                roi_final = roi
                            
                            frame[y_min:y_max, x_min:x_max] = roi_final
                            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                            
                            # Preprocessing & Prediksi
                            gray = cv2.cvtColor(roi_final, cv2.COLOR_BGR2GRAY)
                            enhanced = self.clahe.apply(gray)
                            ai_input = cv2.resize(enhanced, (IMG_SIZE, IMG_SIZE))
                            self.vision_signal.emit(ai_input)
                            
                            inp = ai_input.reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0
                            if self.model:
                                preds = self.model.predict(inp, verbose=0)
                                conf = np.max(preds)
                                if conf > 0.85:
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
        pygame.mixer.init()
        self.setWindowTitle("HandSpeak AI")
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
                self.model = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
                print("Log: Model loaded successfully.")
            except Exception as e: print(f"Error Model: {e}")

    def init_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        main_layout = QVBoxLayout(central); main_layout.setContentsMargins(30, 20, 30, 30)

        header = QLabel("HANDSPEAK AI"); header.setAlignment(Qt.AlignCenter)
        header.setFont(QFont('Segoe UI', 24, QFont.Bold)); header.setStyleSheet("color: #2c3e50;")
        main_layout.addWidget(header)

        body = QHBoxLayout(); body.setSpacing(25)

        #  LEFT PANEL 
        left_side = QVBoxLayout()
        
        # Dropdown Background Selector
        bg_ctrl = QHBoxLayout()
        bg_ctrl.addWidget(QLabel("Contrast Mode:", styleSheet="font-weight: bold;"))
        self.combo_bg = QComboBox()
        self.combo_bg.addItems(["None (Real)", "Solid White", "Solid Black", "Solid Green"])
        self.combo_bg.currentIndexChanged.connect(self.update_bg_mode)
        self.combo_bg.setStyleSheet("height: 30px; padding-left: 10px; border-radius: 5px; background: white; border: 1px solid #dee2e6;")
        bg_ctrl.addWidget(self.combo_bg)
        left_side.addLayout(bg_ctrl)

        self.v_card = QFrame(); self.v_card.setStyleSheet("background: white; border-radius: 20px; border: 1px solid #dee2e6;")
        v_lay = QVBoxLayout(self.v_card)
        self.video_lbl = QLabel(); self.video_lbl.setFixedSize(640, 480); self.video_lbl.setStyleSheet("background: #000; border-radius: 10px;")
        v_lay.addWidget(self.video_lbl)
        left_side.addWidget(self.v_card)

        self.btn_cam = QPushButton(" AKTIFKAN KAMERA"); self.btn_cam.setCheckable(True)
        self.btn_cam.setStyleSheet("background: #007bff; color: white; height: 55px; border-radius: 15px; font-weight: bold;")
        self.btn_cam.clicked.connect(self.toggle_camera)
        left_side.addWidget(self.btn_cam)
        body.addLayout(left_side)

        #  RIGHT PANEL 
        right_side = QVBoxLayout(); right_side.setSpacing(20)

        info_h = QHBoxLayout()
        self.char_card = QFrame(); self.char_card.setStyleSheet("background: white; border-radius: 20px; border: 1px solid #dee2e6;")
        c_lay = QVBoxLayout(self.char_card)
        c_lay.addWidget(QLabel("HURUF TERDETEKSI", alignment=Qt.AlignCenter, styleSheet="color: #6c757d; font-weight: bold;"))
        self.lbl_char = QLabel("-"); self.lbl_char.setFont(QFont('Segoe UI', 80, QFont.Bold)); self.lbl_char.setStyleSheet("color: #007bff;")
        c_lay.addWidget(self.lbl_char, alignment=Qt.AlignCenter)
        self.lbl_conf = QLabel("Confidence: 0%"); self.lbl_conf.setStyleSheet("color: #28a745; font-weight: bold;")
        c_lay.addWidget(self.lbl_conf, alignment=Qt.AlignCenter)
        info_h.addWidget(self.char_card)

        self.vis_card = QFrame(); self.vis_card.setStyleSheet("background: #212529; border-radius: 20px;")
        self.vis_card.setFixedSize(180, 230)
        vi_lay = QVBoxLayout(self.vis_card)
        vi_lay.addWidget(QLabel("AI VISION", alignment=Qt.AlignCenter, styleSheet="color: #adb5bd; font-size: 10px;"))
        self.lbl_vision = QLabel(); self.lbl_vision.setFixedSize(140, 140)
        vi_lay.addWidget(self.lbl_vision, alignment=Qt.AlignCenter)
        info_h.addWidget(self.vis_card)
        right_side.addLayout(info_h)

        # Sentence Box
        self.res_card = QFrame(); self.res_card.setStyleSheet("background: white; border-radius: 20px; border: 1px solid #dee2e6;")
        res_lay = QVBoxLayout(self.res_card)
        res_lay.addWidget(QLabel("KONFIRMASI HASIL", styleSheet="font-weight: bold; color: #6c757d;"))
        self.text_out = QTextEdit(); self.text_out.setFixedHeight(80); self.text_out.setFont(QFont('Segoe UI', 22)); self.text_out.setReadOnly(True)
        res_lay.addWidget(self.text_out)
        
        btn_h = QHBoxLayout()
        self.btn_del = QPushButton(" HAPUS"); self.btn_del.clicked.connect(self.delete_last_char)
        self.btn_clear = QPushButton(" CLEAR"); self.btn_clear.clicked.connect(self.clear_all)
        self.btn_speak = QPushButton(" BACA"); self.btn_speak.clicked.connect(self.speak_text)
        for b in [self.btn_del, self.btn_clear, self.btn_speak]:
            b.setStyleSheet("background: #f8f9fa; border: 1px solid #dee2e6; height: 40px; border-radius: 10px; font-weight: bold;")
            if b == self.btn_speak: b.setStyleSheet("background: #ffc107; font-weight: bold; height: 40px; border-radius: 10px;")
            btn_h.addWidget(b)
        res_lay.addLayout(btn_h)
        right_side.addWidget(self.res_card)

        # Guide Image (RE-INSERTED)
        self.guide_card = QFrame(); self.guide_card.setStyleSheet("background: white; border-radius: 20px; border: 1px solid #dee2e6;")
        g_lay = QVBoxLayout(self.guide_card)
        g_lay.addWidget(QLabel("SIGN LANGUAGE GUIDE", styleSheet="font-weight: bold; color: #6c757d;", alignment=Qt.AlignCenter))
        self.lbl_guide = QLabel()
        if os.path.exists("assets/sign_guide.png"):
            self.lbl_guide.setPixmap(QPixmap("assets/sign_guide.png").scaled(400, 220, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.lbl_guide.setText("File 'assets/sign_guide.png' tidak ditemukan.")
        g_lay.addWidget(self.lbl_guide, alignment=Qt.AlignCenter)
        right_side.addWidget(self.guide_card)

        body.addLayout(right_side)
        main_layout.addLayout(body)

    def update_bg_mode(self, index):
        modes = ["None", "White", "Black", "Green"]
        colors = [(0,0,0), (255,255,255), (0,0,0), (0,255,0)] # BGR
        if self.worker:
            self.worker.set_bg_mode(modes[index], colors[index])

    def speak_text(self):
        text = self.text_out.toPlainText()
        if text:
            try:
                # 1. Stop dan UNLOAD
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()
                
                pygame.mixer.music.unload() # Melepaskan file temp_speech.mp3 dari memori
                
                # 2. Proses gTTS 
                tts = gTTS(text=text, lang='id')
                filename = "speech_output/temp_speech.mp3"
                
                tts.save(filename)
                
                # 3. Load dan putar kembali
                pygame.mixer.music.load(filename)
                pygame.mixer.music.play()
                print("Log: Suara berhasil diputar.")
            except Exception as e: 
                print(f"TTS Error: {e}")

    def toggle_camera(self):
        if self.btn_cam.isChecked():
            self.btn_cam.setText(" MATIKAN KAMERA"); self.btn_cam.setStyleSheet("background: #dc3545; color: white; height: 55px; border-radius: 15px; font-weight: bold;")
            self.worker = VideoWorker(self.model)
            self.worker.change_pixmap_signal.connect(self.update_main)
            self.worker.vision_signal.connect(self.update_vision)
            self.worker.detection_signal.connect(self.handle_logic)
            # Sinkronisasi mode background saat start
            idx = self.combo_bg.currentIndex()
            self.update_bg_mode(idx)
            self.worker.start()
        else:
            self.btn_cam.setText(" AKTIFKAN KAMERA"); self.btn_cam.setStyleSheet("background: #007bff; color: white; height: 55px; border-radius: 15px; font-weight: bold;")
            if self.worker: self.worker.stop()
            self.video_lbl.clear()

    def handle_logic(self, char, conf):
        self.lbl_char.setText(char); self.lbl_conf.setText(f"Confidence: {conf*100:.1f}%")
        if char != "-" and char != "Y": self.pending_char = char
        if char == "Y" and self.pending_char != "":
            self.current_sentence += self.pending_char
            self.text_out.setText(self.current_sentence)
            self.pending_char = ""

    def delete_last_char(self): self.current_sentence = self.current_sentence[:-1]; self.text_out.setText(self.current_sentence)
    def clear_all(self): self.current_sentence = ""; self.text_out.clear()
    def update_main(self, f): h, w, ch = f.shape; qimg = QImage(f.data, w, h, ch * w, QImage.Format_RGB888).rgbSwapped(); self.video_lbl.setPixmap(QPixmap.fromImage(qimg).scaled(640, 480, Qt.KeepAspectRatio))
    def update_vision(self, v): h, w = v.shape; qimg = QImage(v.data, w, h, w, QImage.Format_Grayscale8); self.lbl_vision.setPixmap(QPixmap.fromImage(qimg).scaled(140, 140, Qt.KeepAspectRatio))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HandSpeakApp(); window.show()
    sys.exit(app.exec_())