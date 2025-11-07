# main_gui.py
import sys, subprocess, os, math, csv
from pathlib import Path
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog,
    QVBoxLayout, QWidget, QHBoxLayout, QProgressBar
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal

class Worker(QThread):
    progress = pyqtSignal(str)
    done = pyqtSignal(str, dict)

    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path

    def run(self):
        self.progress.emit("Running nodule detection...")

        stem = Path(self.image_path).stem
        outdir = Path("data/processed")
        outdir.mkdir(parents=True, exist_ok=True)
        csv_path = outdir / f"{stem}_kpts.csv"
        nod_path = outdir / f"{stem}_nodules_p040.csv"
        overlay_path = outdir / f"{stem}_overlay_p040.png"

        # Run keypoint export
        subprocess.run([
            sys.executable, "src/export_keypoints.py",
            "--image", self.image_path,
            "--out", str(csv_path),
            "--min_area", "70",
            "--max_area", "15000",
            "--min_circ", "0.55",
            "--patch", "128",
            "--exclude_color"
        ], check=False)

        # Run classifier inference
        subprocess.run([
            sys.executable, "src/infer_patch_classifier.py",
            "--image", self.image_path,
            "--kpts_csv", str(csv_path),
            "--model", "models/patch_clf_mobilenet.pt",
            "--out_overlay", str(overlay_path),
            "--min_prob", "0.40",
            "--save_csv", str(nod_path)
        ], check=False)

        # Compute coverage
        total_px = 0
        n_count = 0
        with open(nod_path) as f:
            for r in csv.DictReader(f):
                d = float(r["size"])
                total_px += math.pi * (d/2)**2
                n_count += 1

        img = cv2.imread(self.image_path)
        h, w = img.shape[:2]
        coverage = 100 * total_px / (h * w)
        stats = {"count": n_count, "coverage": coverage, "overlay": str(overlay_path)}

        self.done.emit(self.image_path, stats)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nodule Detector GUI")
        self.setGeometry(100, 100, 1000, 700)

        self.image_label = QLabel("Load a .tif image to begin")
        self.image_label.setAlignment(Qt.AlignCenter)

        self.open_btn = QPushButton("Open Image")
        self.run_btn = QPushButton("Run Analysis")
        self.progress = QLabel("")
        self.stats_label = QLabel("")
        self.progressbar = QProgressBar()

        self.open_btn.clicked.connect(self.open_file)
        self.run_btn.clicked.connect(self.run_analysis)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.progressbar)
        layout.addWidget(self.progress)
        layout.addWidget(self.stats_label)

        btns = QHBoxLayout()
        btns.addWidget(self.open_btn)
        btns.addWidget(self.run_btn)
        layout.addLayout(btns)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.image_path = None

    def open_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Image", "data/raw", "Images (*.tif *.png *.jpg)")
        if file:
            self.image_path = file
            pixmap = QPixmap(file)
            pixmap = pixmap.scaled(700, 500, Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)
            self.stats_label.setText("")
            self.progress.setText("Ready to analyze")

    def run_analysis(self):
        if not self.image_path:
            self.progress.setText("No image selected.")
            return
        self.progressbar.setRange(0, 0)  # Indeterminate
        self.worker = Worker(self.image_path)
        self.worker.progress.connect(self.progress.setText)
        self.worker.done.connect(self.show_results)
        self.worker.start()

    def show_results(self, image_path, stats):
        self.progressbar.setRange(0, 1)
        self.progress.setText("Done!")
        self.stats_label.setText(
            f"Nodules Detected: {stats['count']}\n"
            f"Coverage: {stats['coverage']:.3f}%"
        )

        pixmap = QPixmap(stats["overlay"])
        pixmap = pixmap.scaled(700, 500, Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

