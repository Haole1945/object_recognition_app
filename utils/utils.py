import cv2
import numpy as np
from .classes import CLASS_NAMES  # Import CLASS_NAMES từ utils/classes.py

def draw_detections(img, boxes, scores, class_ids, ids, mask_alpha=0.3, allowed_classes=None, conf_thres=0.4):
    if not boxes or not scores or not class_ids:
        return img

    # Màu sắc cho mỗi class
    colors = np.random.randint(0, 255, size=(80, 3), dtype='uint8')

    # Vẽ bounding box và ID
    for box, score, class_id, id_num in zip(boxes, scores, class_ids, ids):
        if score is not None and score < conf_thres:
            continue

        # Lấy tên class
        class_name = CLASS_NAMES.get(class_id, "Unknown")

        # Kiểm tra nếu class_name là "Unknown" thì bỏ qua, không vẽ
        if class_name == "Unknown":
            continue

        print("Drawing - box:", box, "score:", score, "class_id:", class_id, "id_num:", id_num)
        color = (0, 0, 255)

        # Vẽ bounding box lên img
        x1, y1, x2, y2 = map(int, box)
        if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
            print(f"Bounding box {box} vượt giới hạn ảnh {img.shape[:2]}")
            continue
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Hiển thị nhãn
        if score is not None:
            label = f"{id_num} {class_name} {score:.2f}"  # Sửa định dạng label
        else:
            label = f"{id_num} {class_name} N/A"

        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - label_size[1] - baseline), (x1 + label_size[0], y1), color, -1)
        cv2.putText(img, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return img