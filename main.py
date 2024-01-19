import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
import cv2
import numpy as np
from preprocessing import apply_color_constancy, remove_hair

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Skin Lesion Diagnosis App")
        self.setGeometry(100, 100, 800, 600)
        self.createLayouts()
        self.setCentralWidget(self.centralWidget)

    def createLayouts(self):
        self.mainLayout = QHBoxLayout()
        self.createLeftPanel()
        self.createRightPanel()
        self.mainLayout.addWidget(self.leftPanel)
        self.mainLayout.addWidget(self.rightPanel)
        self.centralWidget = QWidget()
        self.centralWidget.setLayout(self.mainLayout)

    def createLeftPanel(self):
        self.leftPanel = QWidget()
        self.leftPanelLayout = QVBoxLayout()
        
        self.uploadButton = QPushButton("Upload Image")
        self.uploadButton.clicked.connect(self.openFileDialog)
        
        self.imageLabel = QLabel("Image will be displayed here")
        self.imageLabel.setAlignment(Qt.AlignCenter)
        
        self.processButton = QPushButton("Process Image")
        self.processButton.clicked.connect(self.processImage)
        
        self.leftPanelLayout.addWidget(self.processButton)
        self.leftPanelLayout.addWidget(self.uploadButton)
        self.leftPanelLayout.addWidget(self.imageLabel)
        
        self.leftPanel.setLayout(self.leftPanelLayout)

    def createRightPanel(self):
        self.rightPanel = QWidget()
        self.rightPanelLayout = QVBoxLayout()
        self.diagnosisLabel = QLabel("Diagnosis will be displayed here")
        self.diagnosisLabel.setAlignment(Qt.AlignCenter)
        self.rightPanelLayout.addWidget(self.diagnosisLabel)
        self.rightPanel.setLayout(self.rightPanelLayout)

    def openFileDialog(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.jpg *.png)", options=options)
        if fileName:
            try:
                if not fileName.lower().endswith(('.png', '.jpg', '.jpeg')):
                    raise ValueError("File format not supported. Please select a JPG or PNG file.")
                self.currentImagePath = fileName  # Set the current image path
                pixmap = QPixmap(fileName)
                self.imageLabel.setPixmap(pixmap.scaled(self.imageLabel.width(), self.imageLabel.height(), Qt.KeepAspectRatio))
            except Exception as e:
                self.imageLabel.setText(str(e))
                self.currentImagePath = None
    
    def processImage(self):
        if self.currentImagePath:  # Make sure there is an image loaded
            # Load the image
            image = cv2.imread(self.currentImagePath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Apply color constancy
            color_constancy_img = apply_color_constancy(image)

            # Apply hair removal
            hair_removed_img = remove_hair(color_constancy_img)

            # Update UI with the final preprocessed image
            self.updateImageLabel(hair_removed_img)

    def updateImageLabel(self, cvImg):
        """Converts a CV image to QImage and updates the image label."""
        height, width, channel = cvImg.shape
        bytesPerLine = 3 * width
        qImg = QImage(cvImg.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)

        # Resize and display the image
        self.imageLabel.setPixmap(pixmap.scaled(self.imageLabel.width(), self.imageLabel.height(), Qt.KeepAspectRatio))
        self.imageLabel.setAlignment(Qt.AlignCenter)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
