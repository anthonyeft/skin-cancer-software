import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from preprocessing import apply_color_constancy
from model.model import caformer_b36

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog, QStackedLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer

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

class SubmissionPage(QWidget):
    def __init__(self):
        super().__init__()
        self.currentImagePath = None
        self.initUI()

    def initUI(self):
        self.uploadButton = QPushButton("Upload Image")
        self.uploadButton.clicked.connect(self.openFileDialog)
        
        self.imageLabel = QLabel("Image will be displayed here")
        self.imageLabel.setAlignment(Qt.AlignCenter)
        
        self.submitButton = QPushButton("Submit")
        self.submitButton.clicked.connect(self.switchToLoading)
        
        layout = QVBoxLayout()

        layout.addWidget(self.submitButton)
        layout.addWidget(self.uploadButton)
        layout.addWidget(self.imageLabel)

        self.setLayout(layout)

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
    
    def switchToLoading(self):
        if self.currentImagePath is not None:
            loadingPage = Loading()
            stackedLayout.addWidget(loadingPage)  # Add loadingPage to the layout
            stackedLayout.setCurrentIndex(3)      # Switch to loadingPage
            # Use a timer to delay the loading process
            QTimer.singleShot(100, lambda: self.startLoadingProcess(loadingPage))  # 100ms delay
        else:
            self.imageLabel.setText("Please upload an image before submitting.")

    def startLoadingProcess(self, loadingPage):
        loadingPage.loadModel()
        loadingPage.processImage(self.currentImagePath)
    
class Loading(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        print("image processed")
    
    def initUI(self):
        self.loadingLabel = QLabel("Loading...")
        self.loadingLabel.setAlignment(Qt.AlignCenter)
        
        layout = QVBoxLayout()
        
        layout.addWidget(self.loadingLabel)

        self.setLayout(layout)
    
    def loadModel(self):
        self.model = caformer_b36(num_classes=7)
        weights_path = 'D:/weights/caformer_b36.pth'
        self.model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        self.model.eval()

    def processImage(self, image_path):
        # Define the transformations to be applied to the image
        test_transform = A.Compose([
            A.Resize(384, 384),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Load and preprocess the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        color_constancy_img = apply_color_constancy(image)

        # Apply test_transform to the image
        transformed = test_transform(image=color_constancy_img)
        input_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension

        # Make prediction
        output = self.model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

        # Map the predicted label number to its corresponding diagnosis name
        diagnosis_name = diagnosis_mapping.get(predicted_class)

        print("Predicted class:", diagnosis_name)
        stackedLayout.setCurrentIndex(2)  # Switch to the LesionReport page
    
class LesionReport(QWidget):
    def __init__(self, diagnosis):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.diagnosisLabel = QLabel("Diagnosis: ")
        self.diagnosisLabel.setAlignment(Qt.AlignCenter)
        
        backButton = QPushButton("Back to Main")
        backButton.clicked.connect(self.goBackToMain)
        
        layout = QVBoxLayout()
        
        layout.addWidget(self.diagnosisLabel)
        layout.addWidget(backButton)

        self.setLayout(layout)

    def goBackToMain(self):
        stackedLayout.setCurrentIndex(1)  # Switch back to the SubmissionPage

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    mainWindow.setWindowTitle("Skin Lesion Diagnosis App")
    mainWindow.setGeometry(100, 100, 1200, 800)
    
    stackedLayout = QStackedLayout()
    
    introPage = IntroductionPage(lambda layout: stackedLayout.setCurrentIndex(1))
    submissionPage = SubmissionPage()
    reportPage = LesionReport()
    
    stackedLayout.addWidget(introPage)
    stackedLayout.addWidget(submissionPage)
    stackedLayout.addWidget(reportPage)
    
    centralWidget = QWidget()
    centralWidget.setLayout(stackedLayout)
    mainWindow.setCentralWidget(centralWidget)
    
    mainWindow.showMaximized()
    sys.exit(app.exec_())