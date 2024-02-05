import sys

from utils.process_image import processImage

from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog, QStackedLayout, QSizePolicy, QSpacerItem, QProgressBar
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QTimer, pyqtSignal

from utils.styles import STYLESHEET

class IntroductionPage(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        introLabel = QLabel("Welcome to the Skin Cancer Diagnosis Demo Application.")

        startButton = QPushButton("Start")
        startButton.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        startButton.clicked.connect(self.switchToMain)

        layout = QVBoxLayout()
        layout.addStretch(10)
        layout.addWidget(introLabel, alignment=Qt.AlignCenter)
        layout.addStretch(1)
        layout.addWidget(startButton, alignment=Qt.AlignCenter)
        layout.addStretch(10)
        self.setLayout(layout)

    def switchToMain(self):
        stackedLayout.setCurrentIndex(1)


class SubmissionPage(QWidget):
    def __init__(self):
        super().__init__()
        self.currentImagePath = None
        self.initUI()

    def initUI(self):
        self.uploadButton = QPushButton("Upload Image")
        self.uploadButton.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.uploadButton.clicked.connect(self.openFileDialog)
        
        self.imageLabel = QLabel("Image will be displayed here")
        self.imageLabel.setAlignment(Qt.AlignCenter)
        
        self.submitButton = QPushButton("Submit")
        self.submitButton.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.submitButton.clicked.connect(self.switchToLoading)
        
        spacer = QSpacerItem(20, 200, QSizePolicy.Minimum, QSizePolicy.Fixed)

        layout = QVBoxLayout()
        layout.addItem(spacer)
        layout.addWidget(self.uploadButton, alignment=Qt.AlignHCenter)
        layout.addWidget(self.imageLabel)
        layout.addWidget(self.submitButton, alignment=Qt.AlignHCenter)
        layout.addItem(spacer)

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
            self.loadingPage = Loading()
            stackedLayout.addWidget(self.loadingPage)  # Add loadingPage to the layout
            stackedLayout.setCurrentIndex(stackedLayout.count() - 1)      # Switch to loadingPage
            # Use a timer to delay the loading process
            QTimer.singleShot(100, lambda: self.startLoadingProcess())  # 100ms delay
        else:
            self.imageLabel.setText("Please upload an image before submitting.")

    def startLoadingProcess(self):
        diagnosis = processImage(self.currentImagePath)
        self.loadingPage.loadingComplete.connect(lambda: self.switchToReport(diagnosis))

    def switchToReport(self, diagnosis):
        reportPage = LesionReport(diagnosis)
        # Remove old LesionReport if it exists
        if stackedLayout.count() > 3:
            oldReportPage = stackedLayout.widget(3)
            stackedLayout.removeWidget(oldReportPage)
            oldReportPage.deleteLater()

        stackedLayout.addWidget(reportPage)  # Add new reportPage to the layout

        QTimer.singleShot(1000, lambda: stackedLayout.setCurrentIndex(stackedLayout.count() - 1))  # Delay the switch to show 100% progress


class Loading(QWidget):
    loadingComplete = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.initUI()
        self.startProgressBar()
    
    def initUI(self):
        self.loadingLabel = QLabel("Loading...")
        self.loadingLabel.setAlignment(Qt.AlignCenter)
        
        self.progressBar = QProgressBar(self)
        self.progressBar.setAlignment(Qt.AlignCenter)
        self.progressBar.setFixedWidth(1000)
        self.progressBar.setRange(0, 100)

        layout = QVBoxLayout()
        layout.addStretch(1)
        layout.addWidget(self.loadingLabel)
        layout.addStretch(1)
        layout.addWidget(self.progressBar, alignment=Qt.AlignHCenter)
        layout.addStretch(1)
        self.setLayout(layout)

    def startProgressBar(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateProgressBar)
        self.timer.start(50)  # Update every 100 ms

        self.progress = 0

    def updateProgressBar(self):
        self.progress += 1  # Increment the progress
        self.progressBar.setValue(self.progress)

        if self.progress >= 100:
            self.timer.stop()  # Stop the timer when progress is complete
            self.loadingComplete.emit()  # Emit the loadingComplete signal
    
class LesionReport(QWidget):
    def __init__(self, diagnosis):
        super().__init__()
        self.diagnosis = diagnosis
        self.initUI()
        
    def initUI(self):
        self.diagnosisLabel = QLabel(f"Diagnosis: {self.diagnosis}")
        self.diagnosisLabel.setAlignment(Qt.AlignCenter)
        
        backButton = QPushButton("Back to Submission Page")
        backButton.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        backButton.clicked.connect(self.goBackToMain)
        
        layout = QVBoxLayout()
        layout.addStretch(4)
        layout.addWidget(self.diagnosisLabel)
        layout.addStretch(1)
        layout.addWidget(backButton, alignment=Qt.AlignHCenter)
        layout.addStretch(3)

        self.setLayout(layout)

    def goBackToMain(self):
        stackedLayout.setCurrentIndex(1)  # Switch back to the SubmissionPage


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    mainWindow = QMainWindow()
    mainWindow.setWindowTitle("Skin Lesion Diagnosis App")
    mainWindow.setGeometry(100, 100, 1200, 800)
    
    stackedLayout = QStackedLayout()
    
    introPage = IntroductionPage()
    submissionPage = SubmissionPage()
    
    stackedLayout.addWidget(introPage)
    stackedLayout.addWidget(submissionPage)
    
    centralWidget = QWidget()
    centralWidget.setLayout(stackedLayout)
    mainWindow.setCentralWidget(centralWidget)
    
    mainWindow.showMaximized()
    sys.exit(app.exec_())