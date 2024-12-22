# import cv2
# import time
# import numpy as np
# from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
# from PyQt5.QtGui import QImage, QPixmap
# from PyQt5.QtWidgets import (QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
#                              QPushButton, QFileDialog, QSlider, QSizePolicy, QProgressBar)
# from utils.detectors import YOLOv8Detector
# from utils.utils import draw_detections
# import qtawesome as qta

# class VideoThread(QThread):
#     change_pixmap_signal = pyqtSignal(np.ndarray)
#     detection_results_signal = pyqtSignal(list, list, list)

#     def __init__(self, detector, video_path=0):
#         super().__init__()
#         self.detector = detector
#         self._run_flag = True
#         self.video_path = video_path
#         self.cap = None

#     def run(self):
#         self.cap = cv2.VideoCapture(self.video_path)
#         if not self.cap.isOpened():
#             print("Error: Could not open video stream.")
#             return

#         while self._run_flag:
#             ret, frame = self.cap.read()
#             if ret:
#                 boxes, scores, class_ids = self.detector(frame)
#                 boxes = boxes.tolist()
#                 scores = scores.tolist()
#                 class_ids = class_ids.tolist()
#                 self.detection_results_signal.emit(boxes, scores, class_ids)
#                 self.change_pixmap_signal.emit(frame)
#             # Introduce a small delay to avoid 100% CPU usage
#             time.sleep(0.01)

#         self.cap.release()

#     def stop(self):
#         """Sets run flag to False and waits for thread to finish"""
#         self._run_flag = False
#         self.wait()

# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Object Detection App")
#         self.setMinimumSize(800, 600)

#         # Initialize YOLOv8 detector
#         self.detector = YOLOv8Detector("weights/yolov8n.pt")
#         self.video_thread = None
#         self.video_path = None

#         # UI elements
#         self.central_widget = QWidget()
#         self.setCentralWidget(self.central_widget)

#         self.main_layout = QVBoxLayout(self.central_widget)
#         self.main_layout.setContentsMargins(10, 10, 10, 10)
#         self.main_layout.setSpacing(10)

#         self.video_label = QLabel("No video loaded")
#         self.video_label.setAlignment(Qt.AlignCenter)
#         self.video_label.setMinimumHeight(360)
#         self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
#         self.main_layout.addWidget(self.video_label)

#         self.progress_bar = QProgressBar()
#         self.progress_bar.setVisible(False)
#         self.main_layout.addWidget(self.progress_bar)

#         self.control_layout = QHBoxLayout()
#         self.control_layout.setContentsMargins(0, 0, 0, 0)
#         self.control_layout.setSpacing(10)

#         self.open_button = QPushButton(qta.icon('fa5s.folder-open'), "Open Video")
#         self.open_button.setIconSize(QSize(24, 24))
#         self.open_button.clicked.connect(self.open_file)
#         self.control_layout.addWidget(self.open_button)

#         self.webcam_button = QPushButton(qta.icon('fa5s.camera'), "Open Webcam")
#         self.webcam_button.setIconSize(QSize(24, 24))
#         self.webcam_button.clicked.connect(self.open_webcam)
#         self.control_layout.addWidget(self.webcam_button)

#         self.play_button = QPushButton(qta.icon('fa5s.play'), "Play")
#         self.play_button.setIconSize(QSize(24, 24))

#         self.play_button.setEnabled(False)
#         self.play_button.clicked.connect(self.play_video)
#         self.control_layout.addWidget(self.play_button)

#         self.stop_button = QPushButton(qta.icon('fa5s.stop'), "Stop")
#         self.stop_button.setIconSize(QSize(24, 24))
#         self.stop_button.setEnabled(False)
#         self.stop_button.clicked.connect(self.stop_video)
#         self.control_layout.addWidget(self.stop_button)

#         self.main_layout.addLayout(self.control_layout)

#         # Confidence threshold slider
#         self.conf_layout = QHBoxLayout()
#         self.conf_label = QLabel("Confidence Threshold:")
#         self.conf_layout.addWidget(self.conf_label)
#         self.conf_slider = QSlider(Qt.Horizontal)
#         self.conf_slider.setRange(0, 100)
#         self.conf_slider.setValue(40)  # Default value
#         self.conf_slider.valueChanged.connect(self.update_confidence_threshold)
#         self.conf_layout.addWidget(self.conf_slider)
#         self.conf_value_label = QLabel("0.40")
#         self.conf_layout.addWidget(self.conf_value_label)
#         self.main_layout.addLayout(self.conf_layout)

#         # FPS display
#         self.fps_label = QLabel("FPS: 0")
#         self.main_layout.addWidget(self.fps_label)

#         # Initialize prev_frame_time
#         self.prev_frame_time = time.time()
#         self.new_frame_time = 0
#         self.last_fps_update_time = time.time() # time of last update FPS

#         # Timer for updating FPS
#         self.fps_timer = QTimer(self)
#         self.fps_timer.timeout.connect(self.update_fps)
#         self.fps_timer.start(1000)  # Update every second

#     def open_file(self):
#         file_name, _ = QFileDialog.getOpenFileName(self, "Open Video File", "",
#                                                    "Videos (*.mp4 *.avi *.mov);;All Files (*)")
#         if file_name:
#             self.progress_bar.setVisible(True)
#             self.progress_bar.setRange(0, 0)  # Indeterminate
#             self.load_video(file_name)
#             self.progress_bar.setVisible(False)

#     def open_webcam(self):
#         self.load_video(0)  # 0 represents the default webcam

#     def load_video(self, video_path):
#         self.video_path = video_path
#         if self.video_thread is not None:
#             self.video_thread.stop()

#         self.video_thread = VideoThread(self.detector, video_path)
#         self.video_thread.change_pixmap_signal.connect(self.update_image)
#         self.video_thread.detection_results_signal.connect(self.draw_results)
#         self.video_thread.start()
#         self.play_button.setEnabled(True)
#         self.stop_button.setEnabled(True)
#         self.play_button.setIcon(qta.icon('fa5s.pause'))

#     def play_video(self):
#         if self.video_thread is not None:
#             if self.video_thread.isRunning():
#                 # Pause the video
#                 self.video_thread.stop()
#                 self.play_button.setIcon(qta.icon('fa5s.play'))
#             else:
#                 # Resume the video
#                 self.video_thread = VideoThread(self.detector, self.video_path)
#                 self.video_thread.change_pixmap_signal.connect(self.update_image)
#                 self.video_thread.detection_results_signal.connect(self.draw_results)
#                 self.video_thread.start()
#                 self.play_button.setIcon(qta.icon('fa5s.pause'))

#     def stop_video(self):
#         if self.video_thread is not None:
#             self.video_thread.stop()
#             self.video_thread = None
#             self.play_button.setIcon(qta.icon('fa5s.play'))
#             self.play_button.setEnabled(False)
#             self.stop_button.setEnabled(False)

#     def update_image(self, cv_img):
#         """Updates the image_label with a new opencv image"""
#         self.new_frame_time = time.time()
#         qt_img = self.convert_cv_qt(cv_img)
#         self.video_label.setPixmap(qt_img)

#     def draw_results(self, boxes, scores, class_ids):
#         # Get the current frame from the video label
#         pixmap = self.video_label.pixmap()
#         if pixmap is not None:
#             img = self.pixmap_to_array(pixmap)
#             combined_img = draw_detections(img, boxes, scores, class_ids,
#                                            mask_alpha=0.3, allowed_classes=['person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle'])

#             # Convert back to QImage and update the label
#             qt_img = self.convert_cv_qt(combined_img)
#             self.video_label.setPixmap(qt_img)

#     def convert_cv_qt(self, cv_img):
#         """Convert from an opencv image to QPixmap"""
#         rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
#         h, w, ch = rgb_image.shape
#         bytes_per_line = ch * w
#         convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
#         p = convert_to_Qt_format.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio)
#         return QPixmap.fromImage(p)

#     def pixmap_to_array(self, pixmap):
#         """Convert QPixmap to numpy array (OpenCV image)"""
#         qimage = pixmap.toImage()
#         buffer = qimage.constBits()
#         buffer.setsize(qimage.byteCount())
#         img = np.array(buffer).reshape(qimage.height(), qimage.width(), 4)  # 4 for RGBA
#         img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
#         return img

#     def update_confidence_threshold(self, value):
#         conf_thres = value / 100.0
#         self.detector.conf_thres = conf_thres
#         self.conf_value_label.setText(f"{conf_thres:.2f}")

#     def update_fps(self):
#         current_time = time.time()
#         # Update FPS only every 0.5 seconds to avoid too frequent updates
#         if current_time - self.last_fps_update_time >= 0.5:
#             time_diff = self.new_frame_time - self.prev_frame_time

#             # Avoid division by zero
#             if time_diff > 0:
#                 fps = 1 / time_diff
#                 self.fps_label.setText(f"FPS: {int(fps)}")
#                 self.last_fps_update_time = current_time

#         self.prev_frame_time = self.new_frame_time

#     def closeEvent(self, event):
#         if self.video_thread is not None:
#             self.video_thread.stop()
#         event.accept()


import cv2
import time
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
                             QPushButton, QFileDialog, QSlider, QSizePolicy, QProgressBar)
from utils.detectors import YOLOv8Detector
from utils.utils import draw_detections
import qtawesome as qta

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

    def run(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print("Error: Could not open video stream.")
            return

        while self._run_flag:
            ret, frame = self.cap.read()
            if ret:
                boxes, scores, class_ids = self.detector(frame)
                # Simple object tracking based on Euclidean distance
                boxes, scores, class_ids, object_ids = self.track_objects(
                    boxes, scores, class_ids, frame
                )

                boxes = (
                    boxes.tolist()
                    if isinstance(boxes, np.ndarray)
                    else boxes
                )
                scores = (
                    scores.tolist()
                    if isinstance(scores, np.ndarray)
                    else scores
                )
                class_ids = (
                    class_ids.tolist()
                    if isinstance(class_ids, np.ndarray)
                    else class_ids
                )
                object_ids = (
                    object_ids.tolist()
                    if isinstance(object_ids, np.ndarray)
                    else object_ids
                )

                print("boxes:", boxes)
                print("scores:", scores)
                print("class_ids:", class_ids)
                print("object_ids:", object_ids)

                self.detection_results_signal.emit(
                    boxes, scores, class_ids, object_ids
                )
                self.change_pixmap_signal.emit(frame)

            # Introduce a small delay to avoid 100% CPU usage
            time.sleep(0.01)

        self.cap.release()

    def track_objects(self, boxes, scores, class_ids, frame):
        object_ids = []
        if len(boxes) > 0:
            # Calculate centers of bounding boxes
            centers = np.array([((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) for box in boxes])

            # If no objects are currently tracked, assign new IDs to all detected objects
            if not self.object_ids:
                for i in range(len(boxes)):
                    self.object_ids[self.next_object_id] = centers[i]
                    object_ids.append(self.next_object_id)
                    self.next_object_id += 1
            else:
                # Calculate distances between new centers and tracked centers
                tracked_centers = np.array(list(self.object_ids.values()))
                distances = np.sqrt(((centers[:, np.newaxis, :] - tracked_centers[np.newaxis, :, :]) ** 2).sum(axis=2))

                # Update tracked objects with new positions
                matched_indices = {}
                for i, center in enumerate(centers):
                    if distances.size > 0:  # Check if distances is not empty
                        min_dist_idx = np.argmin(distances[i, :])
                        if distances[i, min_dist_idx] < 50:
                            # Match found
                            object_id = list(self.object_ids.keys())[min_dist_idx]
                            self.object_ids[object_id] = center
                            object_ids.append(object_id)
                            matched_indices[min_dist_idx] = True
                            distances = np.delete(distances, min_dist_idx, axis=1)
                    else:
                         # Assign new IDs to new objects
                        self.object_ids[self.next_object_id] = centers[i]
                        object_ids.append(self.next_object_id)
                        self.next_object_id += 1

                # Remove lost objects
                object_ids_to_remove = [object_id for object_id in self.object_ids.keys() if object_id not in object_ids]
                for object_id in object_ids_to_remove:
                    self.object_ids.pop(object_id)

                # Assign new IDs to new objects
                for i in range(len(boxes)):
                    if i >= len(object_ids):
                        self.object_ids[self.next_object_id] = centers[i]
                        object_ids.append(self.next_object_id)
                        self.next_object_id += 1

        # Update the last detected boxes for tracking
        self.last_boxes = boxes
        self.last_scores = scores
        self.last_class_ids = class_ids

        return boxes, scores, class_ids, object_ids

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Object Detection App")
        self.setMinimumSize(800, 600)

        # Initialize YOLOv8 detector
        self.detector = YOLOv8Detector("weights/yolov8n.pt")
        self.video_thread = None
        self.video_path = None

        # UI elements
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)

        self.video_label = QLabel("No video loaded")
        self.video_label.setAlignment(Qt.AlignCenter)
        #self.video_label.setMinimumHeight(360)
        self.video_label.setMinimumSize(640, 480)  # Set fixed size
        self.video_label.setMaximumSize(640, 480)  # Set fixed size
        #self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.main_layout.addWidget(self.video_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.main_layout.addWidget(self.progress_bar)

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

        self.play_button = QPushButton(qta.icon('fa5s.play'), "Play")
        self.play_button.setIconSize(QSize(24, 24))

        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.play_video)
        self.control_layout.addWidget(self.play_button)

        self.stop_button = QPushButton(qta.icon('fa5s.stop'), "Stop")
        self.stop_button.setIconSize(QSize(24, 24))
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_video)
        self.control_layout.addWidget(self.stop_button)

        self.main_layout.addLayout(self.control_layout)

        # Confidence threshold slider
        self.conf_layout = QHBoxLayout()
        self.conf_label = QLabel("Confidence Threshold:")
        self.conf_layout.addWidget(self.conf_label)
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(40)  # Default value
        self.conf_slider.valueChanged.connect(self.update_confidence_threshold)
        self.conf_layout.addWidget(self.conf_slider)
        self.conf_value_label = QLabel("0.40")
        self.conf_layout.addWidget(self.conf_value_label)
        self.main_layout.addLayout(self.conf_layout)

        # FPS display
        self.fps_label = QLabel("FPS: 0")
        self.main_layout.addWidget(self.fps_label)

        # Initialize prev_frame_time
        self.prev_frame_time = time.time()
        self.new_frame_time = 0
        self.last_fps_update_time = time.time()  # time of last update FPS

        # Timer for updating FPS
        self.fps_timer = QTimer(self)
        self.fps_timer.timeout.connect(self.update_fps)
        self.fps_timer.start(1000)  # Update every second

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
            if self.video_thread.isRunning():
                # Pause the video
                self.video_thread.stop()
                self.play_button.setIcon(qta.icon('fa5s.play'))
            else:
                # Resume the video
                self.video_thread = VideoThread(self.detector, self.video_path)
                self.video_thread.change_pixmap_signal.connect(self.update_image)
                self.video_thread.detection_results_signal.connect(self.draw_results)
                self.video_thread.start()
                self.play_button.setIcon(qta.icon('fa5s.pause'))

    def stop_video(self):
        if self.video_thread is not None:
            self.video_thread.stop()
            self.video_thread = None
            self.play_button.setIcon(qta.icon('fa5s.play'))
            self.play_button.setEnabled(False)
            self.stop_button.setEnabled(False)

    # def update_image(self, cv_img):
    #     """Updates the image_label with a new opencv image"""
    #     self.new_frame_time = time.time()
    #     qt_img = self.convert_cv_qt(cv_img)
    #     print("qt_img.size():", qt_img.size())
    #     self.video_label.setPixmap(qt_img)

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
            # combined_img = draw_detections(img, boxes, scores, class_ids, ids,
            #                                mask_alpha=0.3, allowed_classes=['person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle'])
            # # Convert back to QImage and update the label
            # qt_img = self.convert_cv_qt(combined_img)

            img = draw_detections(img, boxes, scores, class_ids, ids,
                                           mask_alpha=0.3, allowed_classes=['person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle'])

            # Convert back to QImage and update the label
            qt_img = self.convert_cv_qt(img)
            self.video_label.setPixmap(qt_img)

    # def convert_cv_qt(self, cv_img):
    #     """Convert from an opencv image to QPixmap"""
    #     #rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    #     h, w, ch = cv_img.shape
    #     bytes_per_line = ch * w
    #     convert_to_Qt_format = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_BGR888)
    #     p = convert_to_Qt_format.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio)
    #     return QPixmap.fromImage(p)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        # rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB) # Comment dòng này lại
        h, w, ch = cv_img.shape  # Sửa lại thành thế này
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(
            cv_img.data, w, h, bytes_per_line, QImage.Format_BGR888
        )  # Sửa lại thành thế này
        p = QPixmap.fromImage(convert_to_Qt_format) # Sửa lại thành thế này
        return p

    def pixmap_to_array(self, pixmap):
        """Convert QPixmap to numpy array (OpenCV image)"""
        qimage = pixmap.toImage()
        buffer = qimage.constBits()
        buffer.setsize(qimage.byteCount())
        img = np.array(buffer).reshape(qimage.height(), qimage.width(), 4)  # 4 for RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
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