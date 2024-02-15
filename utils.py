import numpy as np
from PyQt5.QtGui import QImage, QPixmap, QPainter
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsPixmapItem, QGraphicsView
from PyQt5.QtCore import QPropertyAnimation, QEasingCurve, QRectF, QSize, Qt, QTimer, QDateTime


class ABCWidget(QGraphicsView):
    def __init__(self, letter, parent=None):
        super().__init__(parent)
        self.setupGraphics(letter)

    def setupGraphics(self, letter):
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)

        # Define the desired size
        desiredSize = QSize(230, 230)

        # Load and resize the speedometer background
        self.speedometerPixmap = QPixmap(f"ui/static/images/{letter}.png").scaled(desiredSize, Qt.KeepAspectRatio, Qt.SmoothTransformation)
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
        self.targetAngle = score * 260
        self.currentAngle = 0
        self.animationDuration = 2000  # Duration in milliseconds
        self.startTime = QDateTime.currentMSecsSinceEpoch()
        self.easingCurve = QEasingCurve(QEasingCurve.OutCubic)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateNeedleRotation)
        self.timer.start(16)  # 60 FPS

    def updateNeedleRotation(self):
        currentTime = QDateTime.currentMSecsSinceEpoch()
        elapsed = currentTime - self.startTime
        progress = elapsed / self.animationDuration

        if progress < 1.0:
            easedProgress = self.easingCurve.valueForProgress(progress)
            angle = self.currentAngle + (self.targetAngle - self.currentAngle) * easedProgress
            self.needleItem.setRotation(angle)
        else:
            self.needleItem.setRotation(self.targetAngle)
            self.timer.stop()


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


def _weight_mean_color(graph, src, dst, n):
    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}

def merge_mean_color(graph, src, dst):
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] /
                                      graph.nodes[dst]['pixel count'])