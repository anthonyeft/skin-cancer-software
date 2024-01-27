import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from preprocessing import apply_color_constancy
from model.model import caformer_b36

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QFileDialog, QStackedLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

from introduction import IntroductionPage

diagnosis_mapping = {
    0: "Melanoma",
    1: "Benign Melanoctyic Nevi",
    2: "Carcinoma (Basal Cell or Squamous Cell)",
    3: "Actinic Keratoses",
    4: "Benign Keratosis",
    5: "Dermafibroma",
    6: "Vascular Lesion"
}

class MainApplication(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.loadModel()

    def initUI(self):
        self.mainLayout = QHBoxLayout()
        self.createLeftPanel()
        self.createRightPanel()
        self.mainLayout.addWidget(self.leftPanel)
        self.mainLayout.addWidget(self.rightPanel)
        self.setLayout(self.mainLayout)

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
        initialDir = "D:\\Science Fair 2024\\2018_data\\class_separated_data"  # Replace with your desired path
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image", initialDir, "Image Files (*.jpg *.png)", options=options)
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
            # Define the transformations to be applied to the image
            test_transform = A.Compose([
                A.Resize(384, 384),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            
            # Load and preprocess the image
            image = cv2.imread(self.currentImagePath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            color_constancy_img = apply_color_constancy(image)
            
            # Update UI with the preprocessed image
            self.updateImageLabel(color_constancy_img)

            # Apply test_transform to the image
            transformed = test_transform(image=color_constancy_img)
            input_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension

            # Make prediction
            output = self.model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()

            # Map the predicted label number to its corresponding diagnosis name
            diagnosis_name = diagnosis_mapping.get(predicted_class)
            
            # Update the diagnosis label in the UI
            self.diagnosisLabel.setText(f"Diagnosis: {diagnosis_name}")
    
    def loadModel(self):
        self.model = caformer_b36(num_classes=7)
        weights_path = 'D:/weights/caformer_b36.pth'
        self.model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        self.model.eval()

    def prepareModelInput(self, cvImg):
        """Prepare image for model input."""
        # Resize image to the expected input size of the model, 384x384
        resized_img = cv2.resize(cvImg, (384, 384))
        img_tensor = torch.from_numpy(resized_img).float()
        img_tensor = img_tensor.permute(2, 0, 1)  # Convert HWC to CHW
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        return img_tensor

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
    mainWindow = QMainWindow()
    mainWindow.setWindowTitle("Skin Lesion Diagnosis App")
    mainWindow.setGeometry(100, 100, 1200, 800)
    
    stackedLayout = QStackedLayout()
    
    introPage = IntroductionPage(lambda layout: stackedLayout.setCurrentIndex(1))
    mainPage = MainApplication()
    
    stackedLayout.addWidget(introPage)
    stackedLayout.addWidget(mainPage)
    
    centralWidget = QWidget()
    centralWidget.setLayout(stackedLayout)
    mainWindow.setCentralWidget(centralWidget)
    
    mainWindow.showMaximized()
    sys.exit(app.exec_())
