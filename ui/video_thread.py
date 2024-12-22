import cv2
import time
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
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