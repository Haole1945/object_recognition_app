import time
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QFileDialog,
    QSlider,
    QSizePolicy,
    QProgressBar,
    QTableWidget,
    QTableWidgetItem,
)
from PyQt5.QtGui import QIcon
from ui.video_thread import VideoThread
from utils.utils import draw_detections
from utils.database import connect_to_db, create_table, insert_data, close_db_connection
from utils.detectors import YOLOv8Detector
from ui.ui_utils import convert_cv_qt, pixmap_to_array
from utils.classes import CLASS_NAMES, ALLOWED_CLASSES
import qtawesome as qta
import os
import re

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Object Detection App")
        self.setMinimumSize(1500, 600)

        app_icon = QIcon("D:/HaoHaoHao/Hoc Hanh/HK7/Iot/Đồ án/Đồ án cuối kì/object_recognition_app/ui/resources/app_icon.ico")  
        self.setWindowIcon(app_icon)

        # Initialize YOLOv8 detector
        self.detector = YOLOv8Detector("weights/yolov8n.pt")
        self.video_thread = None
        self.video_path = None

        # Database config
        self.db_config = {
            "host": "localhost",
            "user": "root",
            "password": "161003",
            "database": "exportedData",
            "port": 3306,
        }
        self.db_connection = None
        self.db_cursor = None
        self.current_table_name = None
        self.processed_ids = set()  # Set để lưu trữ các ID đã được xử lý

        # UI elements
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)

        # Left Layout
        self.left_layout = QVBoxLayout()

        self.video_label = QLabel("No video loaded")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedSize(640, 480)
        self.left_layout.addWidget(self.video_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.left_layout.addWidget(self.progress_bar)

        # Control Layout (chứa 4 nút)
        self.control_layout = QHBoxLayout()
        self.control_layout.setContentsMargins(0, 0, 0, 0)
        self.control_layout.setSpacing(10)

        self.open_button = QPushButton(qta.icon("fa5s.folder-open"), "Open Video")
        self.open_button.setIconSize(QSize(24, 24))
        self.open_button.clicked.connect(self.open_file)
        self.control_layout.addWidget(self.open_button)

        self.webcam_button = QPushButton(qta.icon("fa5s.camera"), "Open Webcam")
        self.webcam_button.setIconSize(QSize(24, 24))
        self.webcam_button.clicked.connect(self.open_webcam)
        self.control_layout.addWidget(self.webcam_button)

        self.play_button = QPushButton(qta.icon("fa5s.play"), "Play/Pause")
        self.play_button.setIconSize(QSize(24, 24))
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.play_video)
        self.control_layout.addWidget(self.play_button)

        self.stop_button = QPushButton(qta.icon("fa5s.stop"), "Stop")
        self.stop_button.setIconSize(QSize(24, 24))
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_video)
        self.control_layout.addWidget(self.stop_button)

        self.left_layout.addLayout(self.control_layout)

        # Thêm một spacer để tạo khoảng cách giữa control_layout và conf_layout
        spacer_widget = QWidget()
        spacer_widget.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        spacer_widget.setFixedHeight(30)  # Điều chỉnh khoảng cách ở đây
        self.left_layout.addWidget(spacer_widget)

        # Confidence threshold slider (conf_layout)
        self.conf_layout = QHBoxLayout()
        self.conf_label = QLabel("Confidence Threshold:")
        self.conf_layout.addWidget(self.conf_label)

        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(55)  # Default value
        self.conf_slider.valueChanged.connect(self.update_confidence_threshold)
        self.conf_layout.addWidget(self.conf_slider)

        self.conf_value_label = QLabel("0.55")
        self.conf_layout.addWidget(self.conf_value_label)

        # Thêm conf_layout vào left_layout
        self.left_layout.addLayout(self.conf_layout)

        # Thêm một expanding spacer phía dưới conf_layout để đẩy nó lên trên
        self.left_layout.addStretch()

        # Right Layout
        self.right_layout = QVBoxLayout()

        self.processed_video_label = QLabel("Processed video")
        self.processed_video_label.setAlignment(Qt.AlignCenter)
        self.processed_video_label.setFixedSize(640, 480)
        self.right_layout.addWidget(self.processed_video_label)

        # Thêm QTableWidget vào right_layout
        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(3)
        self.table_widget.setHorizontalHeaderLabels(["ID", "Object", "Độ chính xác"])
        self.table_widget.horizontalHeader().setStretchLastSection(True)
        self.table_widget.setEditTriggers(QTableWidget.NoEditTriggers)
        self.right_layout.addWidget(self.table_widget)

        # Add left and right layouts to the main layout
        self.main_layout.addLayout(self.left_layout)
        self.main_layout.addLayout(self.right_layout)

        # Initialize prev_frame_time
        self.prev_frame_time = time.time()
        self.new_frame_time = 0
        self.last_fps_update_time = time.time()  # time of last update FPS

    def open_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video File",
            "",
            "Videos (*.mp4 *.avi *.mov);;All Files (*)",
        )
        if file_name:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate
            self.load_video(file_name)
            self.progress_bar.setVisible(False)

    def open_webcam(self):
        self.load_video(0)  # 0 represents the default webcam

    def load_video(self, video_path):
        self.video_path = video_path
        if self.video_thread is not None:
            self.video_thread.stop()

        # Connect to the database
        self.db_connection, self.db_cursor = connect_to_db(self.db_config)

        # Get the file name (without extension) and create the table name
        if video_path == 0:  # Handle webcam case
            file_name = "webcam"
            self.current_table_name = "ExpDataFromWebcam"
        else:  # Case where video_path is a file path
            file_name = os.path.splitext(os.path.basename(video_path))[0]
            self.current_table_name = (
                "ExpDataFrom" + re.sub(r"[^a-zA-Z0-9]", "", file_name).capitalize()
            )

        # Create a new table (if it doesn't exist)
        create_table(self.db_cursor, self.db_connection, self.current_table_name)

        # Reset processed_ids
        self.processed_ids.clear()

        self.video_thread = VideoThread(self.detector, video_path)
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.detection_results_signal.connect(self.draw_results)
        self.video_thread.start()
        self.play_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.play_button.setIcon(qta.icon("fa5s.pause"))

    def play_video(self):
        if self.video_thread is not None:
            if self.video_thread._paused:
                # Resume the video
                self.video_thread.resume()
                self.play_button.setIcon(qta.icon("fa5s.pause"))
            else:
                # Pause the video
                self.video_thread.pause()
                self.play_button.setIcon(qta.icon("fa5s.play"))

    def stop_video(self):
        if self.video_thread is not None:
            self.video_thread.stop()
            self.video_thread = None
            self.play_button.setIcon(qta.icon("fa5s.play"))
            self.play_button.setEnabled(False)
            self.stop_button.setEnabled(False)

    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        self.new_frame_time = time.time()
        qt_img = convert_cv_qt(cv_img)

        # Scale QPixmap to fit video_label
        scaled_pixmap = qt_img.scaled(self.video_label.size(), Qt.KeepAspectRatio)

        self.video_label.setPixmap(scaled_pixmap)

    def draw_results(self, boxes, scores, class_ids, ids):
        # Get the current frame from the video label
        pixmap = self.video_label.pixmap()
        if pixmap is not None:
            img = pixmap_to_array(pixmap)
            # Draw detections on the image
            img = draw_detections(
                img,
                boxes,
                scores,
                class_ids,
                ids,
                mask_alpha=0.3,
                allowed_classes=ALLOWED_CLASSES,
                conf_thres=self.detector.conf_thres,
            )

            # Convert back to QImage and update the label
            qt_img = convert_cv_qt(img)

            # Scale QPixmap to fit processed_video_label
            scaled_pixmap = qt_img.scaled(
                self.processed_video_label.size(), Qt.KeepAspectRatio
            )

            self.processed_video_label.setPixmap(scaled_pixmap)

            # Lọc các đối tượng có độ tin cậy cao hơn ngưỡng
            filtered_indices = [
                i
                for i, score in enumerate(scores)
                if score >= self.detector.conf_thres
            ]
            filtered_ids = [ids[i] for i in filtered_indices]
            filtered_class_ids = [class_ids[i] for i in filtered_indices]
            filtered_scores = [scores[i] for i in filtered_indices]

            # Cập nhật bảng
            self.table_widget.setRowCount(len(filtered_ids))
            for i, (id_num, class_id, score) in enumerate(
                zip(filtered_ids, filtered_class_ids, filtered_scores)
            ):
                class_name = self.get_class_name(class_id)
                self.table_widget.setItem(i, 0, QTableWidgetItem(str(id_num)))
                self.table_widget.setItem(i, 1, QTableWidgetItem(class_name))
                self.table_widget.setItem(
                    i, 2, QTableWidgetItem(f"{score:.2f}")
                )

                # Thêm dữ liệu vào database
                if (
                    self.db_connection is not None
                    and self.current_table_name is not None
                ):
                    insert_data(
                        self.db_cursor,
                        self.db_connection,
                        self.current_table_name,
                        id_num,
                        class_name,
                        score
                    )

    def get_class_name(self, class_id):
        class_name = CLASS_NAMES.get(class_id, "") 
        return class_name if class_name in ALLOWED_CLASSES else ""

    def update_confidence_threshold(self, value):
        conf_thres = value / 100.0
        self.detector.conf_thres = conf_thres
        self.conf_value_label.setText(f"{conf_thres:.2f}")

    def update_fps(self):
        current_time = time.time()
        # Update FPS only every 0.5 seconds to avoid too frequent updates
        if current_time - self.last_fps_update_time >= 0.5:
            time_diff = self.new_frame_time - self.prev_frame_time

            # Avoid division by zero
            if time_diff > 0:
                fps = 1 / time_diff
                self.fps_label.setText(f"FPS: {int(fps)}")
                self.last_fps_update_time = current_time

        self.prev_frame_time = self.new_frame_time

    def closeEvent(self, event):
        if self.video_thread is not None:
            self.video_thread.stop()

        # Đóng kết nối database
        if self.db_connection is not None and self.db_connection.is_connected():
            close_db_connection(self.db_cursor, self.db_connection)
            print("Database connection closed.")
        event.accept()