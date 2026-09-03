"""PyQt5 desktop composition for static-image and webcam classification."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QCloseEvent, QImage, QPixmap
from PyQt5.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.models.mobilenetv3 import MODEL_ID
from src.ui.history import HistoryRepository
from src.ui.presentation import present_label
from src.ui.workers import (
    CameraInferenceWorker,
    FramePrediction,
    ImageInferenceWorker,
    Predictor,
)


class HistoryWriter(Protocol):
    def add_image_prediction(
        self,
        *,
        image_path: str,
        model_name: str,
        class_id: str,
        confidence: float,
        topk_json: str,
        low_confidence: bool,
    ) -> int: ...


class WasteClassifierWindow(QMainWindow):
    def __init__(
        self,
        predictor: Predictor,
        history: HistoryRepository | HistoryWriter,
        *,
        camera_index: int = 0,
    ) -> None:
        super().__init__()
        self.predictor = predictor
        self.history = history
        self.camera_index = camera_index
        self._image_worker: ImageInferenceWorker | None = None
        self._camera_worker: CameraInferenceWorker | None = None

        self.setWindowTitle("Phân loại rác bằng MobileNetV3")
        self.resize(780, 720)
        self._build_widgets()

    def _build_widgets(self) -> None:
        root = QWidget(self)
        layout = QVBoxLayout(root)
        layout.setSpacing(12)

        title = QLabel("PHÂN LOẠI RÁC – 10 NHÓM")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: 700; color: #0F172A;")
        layout.addWidget(title)

        self.preview_label = QLabel("Chọn ảnh hoặc bật camera để bắt đầu")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(560, 320)
        self.preview_label.setStyleSheet(
            "background: #F1F5F9; border: 2px dashed #94A3B8; "
            "border-radius: 12px; color: #475569;"
        )
        layout.addWidget(self.preview_label, stretch=1)

        actions = QHBoxLayout()
        self.select_image_button = QPushButton("Chọn ảnh")
        self.camera_button = QPushButton("Bật camera")
        self.select_image_button.clicked.connect(self.select_image)
        self.camera_button.clicked.connect(self.toggle_camera)
        actions.addWidget(self.select_image_button)
        actions.addWidget(self.camera_button)
        layout.addLayout(actions)

        self.result_label = QLabel("Chưa có kết quả")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 30px; font-weight: 700;")
        layout.addWidget(self.result_label)

        self.confidence_label = QLabel("Độ tin cậy: —")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.confidence_label)

        self.warning_label = QLabel(
            "Kết quả có độ tin cậy thấp; hãy thử ảnh rõ hơn."
        )
        self.warning_label.setAlignment(Qt.AlignCenter)
        self.warning_label.setStyleSheet(
            "background: #FEF3C7; color: #92400E; padding: 8px; border-radius: 6px;"
        )
        self.warning_label.hide()
        layout.addWidget(self.warning_label)

        top_title = QLabel("Top 3 dự đoán")
        top_title.setStyleSheet("font-weight: 600;")
        layout.addWidget(top_title)
        self.top_three = QListWidget()
        self.top_three.setMaximumHeight(110)
        layout.addWidget(self.top_three)

        self.status_label = QLabel("Sẵn sàng")
        self.status_label.setStyleSheet("color: #475569;")
        layout.addWidget(self.status_label)
        self.setCentralWidget(root)

    def select_image(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn ảnh rác",
            "",
            "Ảnh (*.jpg *.jpeg *.png *.webp *.bmp)",
        )
        if not selected:
            return
        self.start_image_prediction(Path(selected))

    def start_image_prediction(self, image_path: Path) -> None:
        if self._image_worker is not None and self._image_worker.isRunning():
            return
        self.select_image_button.setEnabled(False)
        self.status_label.setText(f"Đang phân loại {image_path.name}…")
        worker = ImageInferenceWorker(self.predictor, image_path)
        self._image_worker = worker
        worker.result_ready.connect(self.handle_result)
        worker.error.connect(self.handle_image_error)
        worker.finished.connect(self._handle_image_finished)
        worker.start()

    def _handle_image_finished(self) -> None:
        self.select_image_button.setEnabled(True)

    def toggle_camera(self) -> None:
        if self._camera_worker is not None and self._camera_worker.isRunning():
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self) -> None:
        if self._camera_worker is not None and self._camera_worker.isRunning():
            return
        worker = CameraInferenceWorker(
            self.predictor,
            camera_index=self.camera_index,
        )
        self._camera_worker = worker
        worker.result_ready.connect(self.handle_result)
        worker.error.connect(self.handle_camera_error)
        worker.finished.connect(self._handle_camera_finished)
        self.camera_button.setText("Tắt camera")
        self.status_label.setText("Đang nhận hình ảnh từ camera…")
        worker.start()

    def stop_camera(self) -> None:
        worker = self._camera_worker
        if worker is not None and worker.isRunning():
            worker.stop()
            worker.wait(1500)
        self.camera_button.setText("Bật camera")
        self.status_label.setText("Đã tắt camera")

    def _handle_camera_finished(self) -> None:
        self.camera_button.setText("Bật camera")

    def handle_result(self, frame_result: FramePrediction) -> None:
        prediction = frame_result.prediction
        presentation = present_label(prediction.top1.class_id)
        self.result_label.setText(
            f"{presentation.icon} {presentation.display_name}"
        )
        self.result_label.setStyleSheet(
            f"font-size: 30px; font-weight: 700; color: {presentation.color};"
        )
        self.confidence_label.setText(
            f"Độ tin cậy: {prediction.top1.probability:.1%}"
        )
        self.warning_label.setVisible(prediction.low_confidence)
        self.top_three.clear()
        for scored in prediction.topk:
            label = present_label(scored.class_id)
            self.top_three.addItem(
                f"{label.icon} {label.display_name}: {scored.probability:.1%}"
            )
        self.show_rgb_frame(frame_result.rgb_frame)
        self.status_label.setText("Đã phân loại xong")
        if frame_result.source_path is not None:
            self.persist_static_result(frame_result)

    def show_rgb_frame(self, rgb_frame: np.ndarray) -> None:
        contiguous = np.ascontiguousarray(rgb_frame, dtype=np.uint8)
        height, width, channels = contiguous.shape
        if channels != 3:
            raise ValueError("RGB frame must have exactly three channels")
        image = QImage(
            contiguous.data,
            width,
            height,
            int(contiguous.strides[0]),
            QImage.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(image).scaled(
            self.preview_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.preview_label.setPixmap(pixmap)

    def persist_static_result(self, frame_result: FramePrediction) -> None:
        prediction = frame_result.prediction
        assert frame_result.source_path is not None
        self.history.add_image_prediction(
            image_path=frame_result.source_path,
            model_name=MODEL_ID,
            class_id=prediction.top1.class_id,
            confidence=prediction.top1.probability,
            topk_json=json.dumps(
                [asdict(scored) for scored in prediction.topk],
                ensure_ascii=False,
            ),
            low_confidence=prediction.low_confidence,
        )

    def handle_image_error(self, message: str) -> None:
        self.status_label.setText(message)
        self.select_image_button.setEnabled(True)

    def handle_camera_error(self, message: str) -> None:
        self.status_label.setText(message)
        self.camera_button.setText("Bật camera")
        self.camera_button.setEnabled(True)
        self.select_image_button.setEnabled(True)

    def closeEvent(self, event: QCloseEvent) -> None:
        camera_worker = self._camera_worker
        if camera_worker is not None and camera_worker.isRunning():
            camera_worker.stop()
            camera_worker.wait(1500)
        image_worker = self._image_worker
        if image_worker is not None and image_worker.isRunning():
            image_worker.requestInterruption()
            image_worker.wait(1500)
        event.accept()
