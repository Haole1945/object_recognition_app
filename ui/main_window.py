import cv2
import time
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
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
                        tracks = self.tracker.update_tracks(
                            detections, frame=frame
                        )  # bbs expected to be a list of আহমেদ, each item being a list with [x1, y1, x2, y2, track_id]

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

                            updated_class_ids.append(class_id)  # Sửa lại thành class_id thay vì gán cứng là 0
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

        # UI elements
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.main_layout = QHBoxLayout(self.central_widget) # Change to QHBoxLayout
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)

        self.left_layout = QVBoxLayout()
        self.right_layout = QVBoxLayout()

        self.video_label = QLabel("No video loaded")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedSize(640, 480)  # Set fixed size
        self.left_layout.addWidget(self.video_label)

        self.processed_video_label = QLabel("Processed video") # New label for processed video
        self.processed_video_label.setAlignment(Qt.AlignCenter)
        self.processed_video_label.setFixedSize(640, 480)  # Set fixed size
        self.right_layout.addWidget(self.processed_video_label)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.right_layout.addWidget(spacer)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.left_layout.addWidget(self.progress_bar)

        spacer = QWidget()
        spacer.setFixedWidth(20)  # Đặt khoảng cách ngang cố định giữa hai video
        self.main_layout.addWidget(spacer)

        self.control_layout = QHBoxLayout()
        self.control_layout.setContentsMargins(0, 0, 0, 0)
        self.control_layout.setSpacing(10)

        self.open_button = QPushButton(qta.icon('fa5s.folder-open'), "Open Video")
        self.open_button.setIconSize(QSize(24, 24))
        self.open_button.clicked.connect(self.open_file)
        self.control_layout.addWidget(self.open_button)

        self.webcam_button = QPushButton(qta.icon('fa5s.camera'), "Open Webcam")
        self.webcam_button.setIconSize(QSize(24, 24))
        self.webcam_button.clicked.connect(self.open_webcam)
        self.control_layout.addWidget(self.webcam_button)

        self.play_button = QPushButton(qta.icon('fa5s.play'), "Play/Pause")
        self.play_button.setIconSize(QSize(24, 24))

        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.play_video)
        self.control_layout.addWidget(self.play_button)

        self.stop_button = QPushButton(qta.icon('fa5s.stop'), "Stop")
        self.stop_button.setIconSize(QSize(24, 24))
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_video)
        self.control_layout.addWidget(self.stop_button)

        self.left_layout.addLayout(self.control_layout)

        # Confidence threshold slider
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
        self.left_layout.addLayout(self.conf_layout)

        # Add left and right layouts to the main layout
        self.main_layout.addLayout(self.left_layout)
        self.main_layout.addLayout(self.right_layout)

        # Initialize prev_frame_time
        self.prev_frame_time = time.time()
        self.new_frame_time = 0
        self.last_fps_update_time = time.time()  # time of last update FPS

    def open_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video File", "",
                                                   "Videos (*.mp4 *.avi *.mov);;All Files (*)")
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

        self.video_thread = VideoThread(self.detector, video_path)
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.detection_results_signal.connect(self.draw_results)
        self.video_thread.start()
        self.play_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.play_button.setIcon(qta.icon('fa5s.pause'))

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
                conf_thres=self.detector.conf_thres
            )

            # Convert back to QImage and update the label
            qt_img = self.convert_cv_qt(img)

            # Scale QPixmap to fit processed_video_label
            scaled_pixmap = qt_img.scaled(
                self.processed_video_label.size(), Qt.KeepAspectRatio
            )

            self.processed_video_label.setPixmap(scaled_pixmap)

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
        event.accept()