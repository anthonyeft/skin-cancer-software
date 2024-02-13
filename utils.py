import numpy as np
from PyQt5.QtGui import QImage, QPixmap, QPainter
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsPixmapItem, QGraphicsView
from PyQt5.QtCore import QPropertyAnimation, QEasingCurve, QRectF, QSize, Qt


class ABCWidget(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupGraphics()

    def setupGraphics(self):
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)

        # Define the desired size
        desiredSize = QSize(200, 200)

        # Load and resize the speedometer background
        self.speedometerPixmap = QPixmap("ui/static/images/A.png").scaled(desiredSize, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.speedometerItem = QGraphicsPixmapItem(self.speedometerPixmap)
        self.scene.addItem(self.speedometerItem)

        # Load and resize the needle
        self.needlePixmap = QPixmap("ui/static/images/needle.png").scaled(desiredSize, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.needleItem = QGraphicsPixmapItem(self.needlePixmap)
        # Adjust the transform origin point based on the resized image
        self.needleItem.setTransformOriginPoint(self.needlePixmap.width() / 2, self.needlePixmap.height() / 2)
        self.scene.addItem(self.needleItem)

        self.setSceneRect(QRectF(self.speedometerPixmap.rect()))

    def animateNeedle(self, score):
        # Map the score (0 to 1) to the angle range (0 to 260 degrees)
        angle = score * 260

        # Create an animation for the rotation
        self.animation = QPropertyAnimation(self.needleItem, b"rotation")
        self.animation.setDuration(2000)  # Duration in milliseconds
        self.animation.setStartValue(0)  # Starting angle
        self.animation.setEndValue(angle)  # Ending angle
        self.animation.setEasingCurve(QEasingCurve.OutCubic)  # Animation effect
        self.animation.start()


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
