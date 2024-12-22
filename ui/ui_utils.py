from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np

def convert_cv_qt(cv_img):
    """Convert from an opencv image to QPixmap"""
    h, w, ch = cv_img.shape
    bytes_per_line = ch * w
    convert_to_Qt_format = QImage(
        cv_img.data, w, h, bytes_per_line, QImage.Format_BGR888
    )
    p = QPixmap.fromImage(convert_to_Qt_format)
    return p

def pixmap_to_array(pixmap):
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