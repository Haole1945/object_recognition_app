import cv2
import time
import numpy as np
import os
import re
import mysql.connector
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap
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
from utils.detectors import YOLOv8Detector
from utils.utils import draw_detections
import qtawesome as qta
from deep_sort_realtime.deepsort_tracker import DeepSort

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    detection_results_signal = pyqtSignal(list, list, list, list)

    def __init__(self, detector, video_path=0):
        super().__init__()
        self.detector = detector
        self._run_flag = True
        self.video_path = video_path
        self.cap = None
        self.object_ids = {}  # Dictionary to track object IDs
        self.next_object_id = 0
        self.tracker = DeepSort(max_age=30)
        self._paused = False

    def run(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print("Error: Could not open video stream.")
            return

        while self._run_flag:
            if not self._paused:
                ret, frame = self.cap.read()
                if ret:
                    boxes, scores, class_ids = self.detector(frame)

                    if boxes is not None:
                        detections = []
                        for i in range(len(boxes)):
                            detections.append(
                                (
                                    [
                                        boxes[i][0],
                                        boxes[i][1],
                                        boxes[i][2] - boxes[i][0],
                                        boxes[i][3] - boxes[i][1],
                                    ],
                                    scores[i],
                                    class_ids[i],
                                )
                            )
                        # Update tracker with detections
                        tracks = self.tracker.update_tracks(detections, frame=frame)

                        # Extract bounding boxes, scores, class IDs, and object IDs from tracks
                        updated_boxes = []
                        updated_scores = []
                        updated_class_ids = []
                        updated_object_ids = []
                        for track in tracks:
                            if not track.is_confirmed():
                                continue
                            track_id = track.track_id
                            ltrb = track.to_ltrb()

                            # Lấy confidence score. Nếu là None thì gán bằng 0.0
                            confidence = track.get_det_conf()
                            if confidence is None:
                                confidence = 0.0

                            bbox = [
                                int(ltrb[0]),
                                int(ltrb[1]),
                                int(ltrb[2]),
                                int(ltrb[3]),
                            ]
                            updated_boxes.append(bbox)
                            updated_scores.append(confidence)
                            class_id = track.get_det_class()  # Lấy class_id từ track
                            if class_id is None:
                                class_id = 0  # Gán class_id mặc định là 0 nếu không lấy được từ track

                            updated_class_ids.append(class_id)
                            updated_object_ids.append(track_id)

                        self.detection_results_signal.emit(
                            updated_boxes,
                            updated_scores,
                            updated_class_ids,
                            updated_object_ids,
                        )
                    else:
                        self.detection_results_signal.emit([], [], [], [])
                    self.change_pixmap_signal.emit(frame)
                else:
                    self._run_flag = False
            # Introduce a small delay to avoid 100% CPU usage
            time.sleep(0.01)

        self.cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

    def pause(self):
        """Pause the video thread"""
        self._paused = True

    def resume(self):
        """Resume the video thread"""
        self._paused = False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Object Detection App")
        self.setMinimumSize(1500, 600)

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
        self.table_widget.setHorizontalHeaderLabels(["ID", "Loại Xe", "Độ Tin Cậy"])
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

    def connect_to_db(self):
        try:
            self.db_connection = mysql.connector.connect(**self.db_config)
            self.db_cursor = self.db_connection.cursor()
            print("Connected to database successfully!")
        except mysql.connector.Error as err:
            print(f"Error connecting to database: {err}")

    def create_table(self, table_name):
        try:
            # Sử dụng IF NOT EXISTS để tránh lỗi nếu bảng đã tồn tại
            self.db_cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id INT PRIMARY KEY,
                    class_name VARCHAR(255),
                    confidence FLOAT
                )
                """
            )
            self.db_connection.commit()
            print(f"Table {table_name} created or already exists.")

            # Nếu bảng đã tồn tại, xóa dữ liệu cũ
            self.db_cursor.execute(f"DELETE FROM {table_name}")
            self.db_connection.commit()
            print(f"Data cleared from existing table {table_name}.")

        except mysql.connector.Error as err:
            print(f"Error creating or clearing table {table_name}: {err}")

    def insert_data(self, table_name, id_num, class_name, confidence):
        if id_num not in self.processed_ids:
            try:
                sql = f"INSERT INTO {table_name} (id, class_name, confidence) VALUES (%s, %s, %s)"
                val = (id_num, class_name, confidence)
                self.db_cursor.execute(sql, val)
                self.db_connection.commit()
                self.processed_ids.add(id_num)  # Thêm ID vào set đã xử lý
                print(
                    f"Data inserted: ID={id_num}, Class={class_name}, Confidence={confidence}"
                )
            except mysql.connector.Error as err:
                print(f"Error inserting data: {err}")

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

        # Kết nối database
        self.connect_to_db()

        # Lấy tên file (không có phần mở rộng) và tạo tên bảng
        if video_path == 0:  # Xử lý trường hợp webcam
            file_name = "webcam"
            self.current_table_name = "ExpDataFromWebcam"
        else: # Trường hợp video_path là đường dẫn file
            file_name = os.path.splitext(os.path.basename(video_path))[0]
            self.current_table_name = (
                "ExpDataFrom" + re.sub(r"[^a-zA-Z0-9]", "", file_name).capitalize()
            )

        # Tạo bảng mới (nếu chưa tồn tại)
        self.create_table(self.current_table_name)

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
        qt_img = self.convert_cv_qt(cv_img)

        # Scale QPixmap to fit video_label
        scaled_pixmap = qt_img.scaled(self.video_label.size(), Qt.KeepAspectRatio)

        self.video_label.setPixmap(scaled_pixmap)

    def draw_results(self, boxes, scores, class_ids, ids):
        # Get the current frame from the video label
        pixmap = self.video_label.pixmap()
        if pixmap is not None:
            img = self.pixmap_to_array(pixmap)
            # Draw detections on the image
            img = draw_detections(
                img,
                boxes,
                scores,
                class_ids,
                ids,
                mask_alpha=0.3,
                allowed_classes=[
                    "person",
                    "car",
                    "truck",
                    "bus",
                    "motorcycle",
                    "bicycle",
                ],
                conf_thres=self.detector.conf_thres,
            )

            # Convert back to QImage and update the label
            qt_img = self.convert_cv_qt(img)

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
                    self.insert_data(
                        self.current_table_name, id_num, class_name, score
                    )

    def get_class_name(self, class_id):
        # Dictionary ánh xạ class_id sang tên class (COCO dataset)
        class_names = {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            4: "airplane",
            5: "bus",
            6: "train",
            7: "truck",
            8: "boat",
            9: "traffic light",
            10: "fire hydrant",
            11: "stop sign",
            12: "parking meter",
            13: "bench",
            14: "bird",
            15: "cat",
            16: "dog",
            17: "horse",
            18: "sheep",
            19: "cow",
            20: "elephant",
            21: "bear",
            22: "zebra",
            23: "giraffe",
            24: "backpack",
            25: "umbrella",
            26: "handbag",
            27: "tie",
            28: "suitcase",
            29: "frisbee",
            30: "skis",
            31: "snowboard",
            32: "sports ball",
            33: "kite",
            34: "baseball bat",
            35: "baseball glove",
            36: "skateboard",
            37: "surfboard",
            38: "tennis racket",
            39: "bottle",
            40: "wine glass",
            41: "cup",
            42: "fork",
            43: "knife",
            44: "spoon",
            45: "bowl",
            46: "banana",
            47: "apple",
            48: "sandwich",
            49: "orange",
            50: "broccoli",
            51: "carrot",
            52: "hot dog",
            53: "pizza",
            54: "donut",
            55: "cake",
            56: "chair",
            57: "couch",
            58: "potted plant",
            59: "bed",
            60: "dining table",
            61: "toilet",
            62: "tv",
            63: "laptop",
            64: "mouse",
            65: "remote",
            66: "keyboard",
            67: "cell phone",
            68: "microwave",
            69: "oven",
            70: "toaster",
            71: "sink",
            72: "refrigerator",
            73: "book",
            74: "clock",
            75: "vase",
            76: "scissors",
            77: "teddy bear",
            78: "hair drier",
            79: "toothbrush",
        }
        allowed_classes = [
            "person",
            "car",
            "truck",
            "bus",
            "motorcycle",
            "bicycle",
        ]
        # Trả về tên class nếu nằm trong allowed_classes, ngược lại trả về chuỗi rỗng
        class_name = class_names.get(class_id, "")
        return class_name if class_name in allowed_classes else ""

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(
            cv_img.data, w, h, bytes_per_line, QImage.Format_BGR888
        )
        p = QPixmap.fromImage(convert_to_Qt_format)
        return p

    def pixmap_to_array(self, pixmap):
        """Convert QPixmap to numpy array (OpenCV image)"""
        qimage = pixmap.toImage()

        # Get image dimensions
        width = qimage.width()
        height = qimage.height()

        # Get pointer to the image data
        ptr = qimage.bits()
        ptr.setsize(qimage.byteCount())

        # Create a NumPy array from the raw image data
        img = np.array(ptr).reshape(height, width, 4)  # 4 for RGBA

        # Convert from BGRA to BGR (remove alpha channel)
        img = img[..., :3]
        # Ensure the array is contiguous in memory
        img = np.ascontiguousarray(img)

        # Convert from RGB to BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

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
            self.db_cursor.close()
            self.db_connection.close()
            print("Database connection closed.")
        event.accept()