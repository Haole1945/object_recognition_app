import cv2
import numpy as np

def draw_detections(img, boxes, scores, class_ids, ids, mask_alpha=0.3, allowed_classes=None, conf_thres=0.4):
    if not boxes or not scores or not class_ids:
        return img

    # Màu sắc cho mỗi class
    colors = np.random.randint(0, 255, size=(80, 3), dtype='uint8')

    # Lấy tên class từ dictionary
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

    # Vẽ bounding box và ID
    for box, score, class_id, id_num in zip(boxes, scores, class_ids, ids):
        if score is not None and score < conf_thres:
            continue

        print("Drawing - box:", box, "score:", score, "class_id:", class_id, "id_num:", id_num)
        color = (0, 0, 255)

        # Vẽ bounding box lên img
        x1, y1, x2, y2 = map(int, box)
        if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
            print(f"Bounding box {box} vượt giới hạn ảnh {img.shape[:2]}")
            continue  # Bỏ qua bounding box không hợp lệ
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Hiển thị nhãn
        if score is not None:
            class_name = class_names.get(
                class_id, "Unknown"
            )  # Lấy tên class từ dictionary, mặc định là "Unknown"
            label = f"{id_num} {class_name} {score:.2f}" # Sửa định dạng label
        else:
            class_name = class_names.get(class_id, "Unknown")
            label = f"{id_num} {class_name} N/A"

        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - label_size[1] - baseline), (x1 + label_size[0], y1), color, -1)
        cv2.putText(img, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return img