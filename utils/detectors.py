import cv2
import numpy as np
from ultralytics import YOLO

class YOLOv8Detector:
    def __init__(self, weights_path, conf_thres=0.4, iou_thres=0.5):
        self.model = YOLO(weights_path)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def __call__(self, img):
        # Resize nếu cần thiết
        img_resized = self.resize_image(img) if img.shape[0] > 640 or img.shape[1] > 640 else img

        # Dự đoán
        results = self.model(img_resized, conf=self.conf_thres, iou=self.iou_thres)

        # Lấy kết quả đầu tiên (giả sử chỉ có một hình ảnh)
        result = results[0]

        # Trích xuất boxes, scores, class_ids
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        return boxes, scores, class_ids

    def resize_image(self, img, target_size=640):
        h, w = img.shape[:2]
        scale = min(target_size / w, target_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h))
        return img_resized