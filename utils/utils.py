# import cv2
# import numpy as np

# def draw_detections(img, boxes, scores, class_ids, mask_alpha=0.3, allowed_classes=None):
#     if not boxes or not scores or not class_ids:
#         return img
#     # Tạo một mask màu ngẫu nhiên cho mỗi class
#     colors = np.random.randint(0, 255, size=(80, 3), dtype='uint8')

#     # Tạo một mask để vẽ
#     mask = np.zeros(img.shape, dtype=np.uint8)
#     mask_b = np.zeros(img.shape, dtype=np.bool_)

#     # Vẽ các hộp giới hạn
#     for box, score, class_id in zip(boxes, scores, class_ids):
#         if allowed_classes is None or int(class_id) in [0, 1, 2, 3, 5, 7]:  # Filter class_ids
#             color = [int(c) for c in colors[class_id]]

#             # Vẽ hộp
#             x1, y1, x2, y2 = box
#             cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

#             # Nhãn
#             label = f"{class_id}: {score:.2f}"
#             label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
#             cv2.rectangle(img, (x1, y1 - label_size[1] - baseline), (x1 + label_size[0], y1), color, -1)
#             cv2.putText(img, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

#             # Vẽ mask
#             mask_b[y1:y2, x1:x2, :] = True
#             mask[y1:y2, x1:x2, :] = color

#     # Áp dụng mask
#     img[mask_b] = cv2.addWeighted(img, 1 - mask_alpha, mask, mask_alpha, 0)[mask_b]
#     return img


import cv2
import numpy as np

def draw_detections(img, boxes, scores, class_ids, ids, mask_alpha=0.3, allowed_classes=None):
    if not boxes or not scores or not class_ids:
        return img

    # Màu sắc cho mỗi class
    colors = np.random.randint(0, 255, size=(80, 3), dtype='uint8')

    # Vẽ bounding box và ID
    for box, score, class_id, id_num in zip(boxes, scores, class_ids, ids):
        print("Drawing - box:", box, "score:", score, "class_id:", class_id, "id_num:", id_num)
        color = [int(c) for c in colors[class_id]]

        # Vẽ bounding box lên img
        # x1, y1, x2, y2 = box
        # cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)

        x1, y1, x2, y2 = map(int, box)
        if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
            print(f"Bounding box {box} vượt giới hạn ảnh {img.shape[:2]}")
            continue  # Bỏ qua bounding box không hợp lệ
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.imwrite("output_with_boxes.jpg", img)  # Lưu ảnh kiểm tra

        # Hiển thị nhãn
        label = f"ID: {id_num} - {class_id}: {score:.2f}"
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - label_size[1] - baseline), (x1 + label_size[0], y1), color, -1)
        cv2.putText(img, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return img  