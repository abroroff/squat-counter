"""Squat Counter — kamera orqali o'tirib-turishni sanaydi.

Ishga tushirish:  .venv/bin/python main.py  [--camera 0]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QMainWindow, QProgressBar, QPushButton,
    QSlider, QVBoxLayout, QWidget,
)

from squat_logic import (
    L_ANKLE, L_HIP, L_KNEE, R_ANKLE, R_HIP, R_KNEE, SquatCounter, State, Update,
)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
MODEL_PATH = Path(__file__).resolve().parent / "models" / "pose_landmarker_full.task"

CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]
LEG_JOINTS = {L_HIP, L_KNEE, L_ANKLE, R_HIP, R_KNEE, R_ANKLE}


def ensure_model() -> Path:
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Poza modeli yuklanmoqda (~9 MB): {MODEL_URL}")
        tmp = MODEL_PATH.with_suffix(".part")
        try:
            urllib.request.urlretrieve(MODEL_URL, tmp)
        except Exception:  # noqa: BLE001
            # python.org Python'i macOS'da ko'pincha SSL sertifikatlarini topolmaydi —
            # tizimdagi curl o'z sertifikatlaridan foydalanadi.
            subprocess.run(["curl", "-fsSL", "-o", str(tmp), MODEL_URL], check=True)
        tmp.rename(MODEL_PATH)
    return MODEL_PATH


def draw_pose(frame: np.ndarray, img: np.ndarray, leg_color: tuple[int, int, int]) -> None:
    h, w = frame.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y, _ in img]
    for a, b in CONNECTIONS:
        if img[a, 2] < 0.5 or img[b, 2] < 0.5:
            continue
        color = leg_color if a in LEG_JOINTS and b in LEG_JOINTS else (230, 230, 230)
        cv2.line(frame, pts[a], pts[b], color, 4, cv2.LINE_AA)
    for i in range(11, 33):
        if img[i, 2] >= 0.5:
            cv2.circle(frame, pts[i], 5, (255, 255, 255), -1, cv2.LINE_AA)


class CameraWorker(QThread):
    frame_ready = Signal(QImage)
    updated = Signal(object)  # Update
    failed = Signal(str)

    def __init__(self, camera: int):
        super().__init__()
        self.camera = camera
        self.counter = SquatCounter()
        self._running = True
        self._reset_requested = False

    def request_reset(self) -> None:
        self._reset_requested = True

    def set_bottom_angle(self, deg: int) -> None:
        self.counter.cfg.bottom_knee_angle = float(deg)

    def stop(self) -> None:
        self._running = False
        self.wait(2000)

    def run(self) -> None:
        try:
            model = ensure_model()
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"Modelni yuklab bo'lmadi: {e}")
            return

        cap = cv2.VideoCapture(self.camera, cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY)
        if not cap.isOpened():
            self.failed.emit(
                "Kamera ochilmadi. System Settings → Privacy & Security → Camera "
                "bo'limida Terminal (yoki IDE) ga ruxsat bering."
            )
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.6,
            min_pose_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        t0 = time.monotonic()
        last_ts = -1
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                if self._reset_requested:
                    self.counter.reset()
                    self._reset_requested = False

                frame = cv2.flip(frame, 1)  # oynadagidek ko'rinsin
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                now = time.monotonic() - t0
                ts = max(int(now * 1000), last_ts + 1)
                last_ts = ts
                res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

                img = world = None
                if res.pose_landmarks:
                    img = np.array([[p.x, p.y, p.visibility] for p in res.pose_landmarks[0]])
                    world = np.array([[p.x, p.y, p.z] for p in res.pose_world_landmarks[0]])

                upd = self.counter.update(now, img, world)

                if img is not None:
                    if upd.fail:
                        color = (60, 60, 235)
                    elif upd.state == State.SQUATTING and upd.depth >= 1.0:
                        color = (80, 210, 80)
                    else:
                        color = (0, 200, 255)
                    draw_pose(frame, img, color)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
                self.frame_ready.emit(qimg)
                self.updated.emit(upd)
        cap.release()


class MainWindow(QMainWindow):
    def __init__(self, camera: int):
        super().__init__()
        self.setWindowTitle("Squat Counter")
        self.resize(1280, 760)

        self.video = QLabel("Kamera ishga tushmoqda...")
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setMinimumSize(640, 480)
        self.video.setStyleSheet("background:#111; color:#aaa;")

        self.count_lbl = QLabel("0")
        self.count_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.count_lbl.setFont(QFont("Helvetica", 120, QFont.Weight.Bold))

        self.rejected_lbl = QLabel("Hisoblanmagan: 0")
        self.state_lbl = QLabel("")
        self.state_lbl.setFont(QFont("Helvetica", 16, QFont.Weight.Bold))
        self.msg_lbl = QLabel("")
        self.msg_lbl.setWordWrap(True)
        self.msg_lbl.setFont(QFont("Helvetica", 18))
        self.msg_lbl.setMinimumHeight(70)

        self.depth_bar = QProgressBar()
        self.depth_bar.setRange(0, 100)
        self.depth_bar.setFormat("Chuqurlik: %p%")

        self.knee_lbl = QLabel("Tizzalar: – / –")

        self.angle_lbl = QLabel()
        self.angle_slider = QSlider(Qt.Orientation.Horizontal)
        self.angle_slider.setRange(70, 120)
        self.angle_slider.setValue(100)
        self.angle_slider.valueChanged.connect(self._on_angle)

        reset_btn = QPushButton("Qaytadan boshlash")
        reset_btn.clicked.connect(lambda: self.worker.request_reset())

        help_lbl = QLabel(
            "• Kameradan 2–3 m uzoqda, butun tanangiz ko'rinadigan qilib turing.\n"
            "• Avval 1 soniya qimirlamay tik turing — sozlanadi.\n"
            "• Ikkala tizza bukilib, tos pastga tushgandagina sanaladi.\n"
            "• Oyoqni ko'tarish, joydan siljish, yarim o'tirish sanalmaydi."
        )
        help_lbl.setWordWrap(True)
        help_lbl.setStyleSheet("color:#888;")

        side = QVBoxLayout()
        side.addWidget(self.count_lbl)
        side.addWidget(self.rejected_lbl)
        side.addSpacing(10)
        side.addWidget(self.state_lbl)
        side.addWidget(self.msg_lbl)
        side.addWidget(self.depth_bar)
        side.addWidget(self.knee_lbl)
        side.addSpacing(10)
        side.addWidget(self.angle_lbl)
        side.addWidget(self.angle_slider)
        side.addWidget(reset_btn)
        side.addStretch()
        side.addWidget(help_lbl)

        side_w = QWidget()
        side_w.setLayout(side)
        side_w.setFixedWidth(340)

        root = QHBoxLayout()
        root.addWidget(self.video, 1)
        root.addWidget(side_w)
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

        self.worker = CameraWorker(camera)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.updated.connect(self._on_update)
        self.worker.failed.connect(self._on_failed)
        self._on_angle(self.angle_slider.value())
        self.worker.start()

    def _on_angle(self, v: int) -> None:
        self.angle_lbl.setText(f"Talab qilinadigan tizza burchagi: ≤ {v}°  (kichikroq = chuqurroq)")
        self.worker.set_bottom_angle(v)

    def _on_frame(self, img: QImage) -> None:
        pix = QPixmap.fromImage(img).scaled(
            self.video.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.video.setPixmap(pix)

    def _on_update(self, u: Update) -> None:
        self.count_lbl.setText(str(u.count))
        self.rejected_lbl.setText(f"Hisoblanmagan: {u.rejected}")
        self.state_lbl.setText(f"Holat: {u.state.value}")
        color = {True: "#2ecc71", False: "#e74c3c", None: "#dddddd"}[u.message_ok]
        self.msg_lbl.setStyleSheet(f"color:{color};")
        self.msg_lbl.setText(u.message)
        self.depth_bar.setValue(int(u.depth * 100))
        if u.knee_l is not None:
            self.knee_lbl.setText(f"Tizzalar: {u.knee_l:.0f}° / {u.knee_r:.0f}°")
        if u.event == "rep":
            QApplication.beep()

    def _on_failed(self, text: str) -> None:
        self.video.setText(text)
        self.video.setWordWrap(True)

    def closeEvent(self, e) -> None:  # noqa: N802
        self.worker.stop()
        super().closeEvent(e)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="kamera indeksi (standart 0)")
    args = parser.parse_args()
    app = QApplication(sys.argv)
    win = MainWindow(args.camera)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
