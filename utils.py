import numpy as np
from PyQt5.QtGui import QImage, QPixmap

def convertArrayToPixmap(array):
    if array is None:
        return None
    if len(array.shape) == 2:  # Grayscale
        height, width = array.shape
        return QPixmap.fromImage(QImage(array.data, width, height, width, QImage.Format_Grayscale8))
    elif len(array.shape) == 3:  # Color image
        height, width, channels = array.shape
        bytesPerLine = channels * width
        return QPixmap.fromImage(QImage(array.data, width, height, bytesPerLine, QImage.Format_RGB888))
    else:
        raise ValueError("Unsupported array shape for QPixmap conversion.")
