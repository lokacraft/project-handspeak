import os
import sys
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFrame)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont

# --- KONFIGURASI MODEL ---
MODEL_PATH = "models/fer_emotion_model_v1.keras"
# BERDASARKAN ERROR: Model Anda mengharapkan input 64x64
IMG_SIZE = 64 
EMOTION_LABELS = ['Bengong', 'Bingung', 'Positif', 'Pusing']

class FERWorker(QThread):
    """Thread untuk memproses video agar GUI tetap responsif."""
    change_pixmap_signal = pyqtSignal(np.ndarray)
    emotion_signal = pyqtSignal(str, float)

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.running = True
        
        # Inisialisasi MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.6
        )

    def stop(self):
        self.running = False
        self.wait()

    def run(self):
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Balik frame agar seperti cermin
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # Konversi ke RGB untuk MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.detector.process(rgb_frame)

            label_text = "-"
            confidence_val = 0.0

            if results.detections:
                for detection in results.detections:
                    # Ambil koordinat kotak wajah
                    bboxC = detection.location_data.relative_bounding_box
                    xmin = int(bboxC.xmin * w)
                    ymin = int(bboxC.ymin * h)
                    width = int(bboxC.width * w)
                    height = int(bboxC.height * h)

                    # Pastikan koordinat di dalam batas frame
                    xmin, ymin = max(0, xmin), max(0, ymin)
                    xmax, ymax = min(w, xmin + width), min(h, ymin + height)

                    # Crop area wajah (ROI)
                    face_roi = frame[ymin:ymax, xmin:xmax]

                    if face_roi.size != 0:
                        try:
                            # Preprocessing untuk Model FER (64x64x3)
                            face_input = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
                            face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)
                            
                            # Normalisasi & Reshape ke (1, 64, 64, 3)
                            face_input = np.expand_dims(face_input, axis=0)
                            face_input = face_input.astype('float32') / 255.0
                            
                            # Prediksi Emosi
                            if self.model:
                                # Menggunakan model(input) lebih cepat dan stabil untuk inference real-time
                                preds = self.model(face_input, training=False).numpy()
                                idx = np.argmax(preds)
                                label_text = EMOTION_LABELS[idx]
                                confidence_val = preds[0][idx]

                                # Visualisasi Kotak & Label
                                color = (0, 255, 0) # Hijau
                                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                                cv2.putText(frame, f"{label_text} ({confidence_val*100:.1f}%)", 
                                            (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                            0.7, color, 2)
                        except Exception as e:
                            print(f"Prediction Error: {e}")

            self.emotion_signal.emit(label_text, confidence_val)
            self.change_pixmap_signal.emit(frame)

        cap.release()

class FERApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Facial Emotion Recognition (FER)")
        self.setMinimumSize(1000, 750)
        self.setStyleSheet("background-color: #121212; color: white;")
        
        self.model = None
        self.worker = None
        
        self.load_fer_model()
        self.init_ui()

    def load_fer_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                # Load model dengan compile=False untuk menghindari error optimizer/metrics
                self.model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                print("Log: Model FER berhasil dimuat dengan target input 64x64.")
            except Exception as e:
                print(f"Error Model: {e}")
        else:
            print(f"Peringatan: File {MODEL_PATH} tidak ditemukan.")

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Header
        header = QLabel("FACIAL EMOTION RECOGNITION (64x64)")
        header.setAlignment(Qt.AlignCenter)
        header.setFont(QFont('Segoe UI', 20, QFont.Bold))
        header.setStyleSheet("margin-bottom: 10px; color: #00ff88;")
        layout.addWidget(header)

        # Main Content
        content_layout = QHBoxLayout()

        # Video Frame
        self.video_container = QFrame()
        self.video_container.setStyleSheet("border: 2px solid #333; border-radius: 15px; background: black;")
        video_vbox = QVBoxLayout(self.video_container)
        self.video_label = QLabel("Kamera Mati")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedSize(640, 480)
        video_vbox.addWidget(self.video_label)
        content_layout.addWidget(self.video_container)

        # Info Panel
        info_panel = QVBoxLayout()
        
        # Card Emosi
        self.emotion_card = QFrame()
        self.emotion_card.setStyleSheet("background: #1e1e1e; border-radius: 20px; padding: 20px; border: 2px solid #00ff88;")
        emo_layout = QVBoxLayout(self.emotion_card)
        
        emo_layout.addWidget(QLabel("STATUS EMOSI:", styleSheet="color: #aaa; font-size: 13px; font-weight: bold;"))
        self.lbl_emotion = QLabel("-")
        self.lbl_emotion.setFont(QFont('Segoe UI', 32, QFont.Bold))
        self.lbl_emotion.setStyleSheet("color: #00ff88; margin: 10px 0;")
        self.lbl_emotion.setAlignment(Qt.AlignCenter)
        emo_layout.addWidget(self.lbl_emotion)

        self.lbl_conf = QLabel("Confidence: 0%")
        self.lbl_conf.setAlignment(Qt.AlignCenter)
        self.lbl_conf.setStyleSheet("font-size: 16px; color: #adb5bd;")
        emo_layout.addWidget(self.lbl_conf)
        
        info_panel.addWidget(self.emotion_card)
        info_panel.addStretch()

        # Tombol Kontrol
        self.btn_camera = QPushButton("NYALAKAN KAMERA")
        self.btn_camera.setCheckable(True)
        self.btn_camera.setFixedHeight(60)
        self.btn_camera.setCursor(Qt.PointingHandCursor)
        self.btn_camera.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                border-radius: 12px;
                font-weight: bold;
                font-size: 16px;
                color: white;
            }
            QPushButton:hover {
                background-color: #0069d9;
            }
            QPushButton:checked {
                background-color: #dc3545;
            }
        """)
        self.btn_camera.clicked.connect(self.toggle_camera)
        info_panel.addWidget(self.btn_camera)

        content_layout.addLayout(info_panel)
        layout.addLayout(content_layout)

    def toggle_camera(self):
        if self.btn_camera.isChecked():
            self.btn_camera.setText("MATIKAN KAMERA")
            self.worker = FERWorker(self.model)
            self.worker.change_pixmap_signal.connect(self.update_video)
            self.worker.emotion_signal.connect(self.update_info)
            self.worker.start()
        else:
            self.btn_camera.setText("NYALAKAN KAMERA")
            if self.worker:
                self.worker.stop()
            self.video_label.setText("Kamera Mati")
            self.video_label.clear()

    def update_video(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.video_label.setPixmap(qt_img)

    def update_info(self, emotion, confidence):
        self.lbl_emotion.setText(emotion.upper())
        self.lbl_conf.setText(f"Confidence: {confidence*100:.1f}%")

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = FERApp()
    window.show()
    sys.exit(app.exec_())