from ui_sidebar import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow, QPushButton, QWidget, QVBoxLayout, QLabel, QFileDialog, QSizePolicy, QProgressBar
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from utils.process_image import processImage

class Sidebar(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle('Demo Application')

        self.icon_name_sidebar_widget.setHidden(True)

        self.dashboard_button.clicked.connect(self.switch_to_dashboard)
        self.dashboard_logo_button.clicked.connect(self.switch_to_dashboard)

        self.patients_button.clicked.connect(self.switch_to_patients)
        self.patients_logo_button.clicked.connect(self.switch_to_patients)
        
        self.quick_scan_button.clicked.connect(self.switch_to_quick_scan)
        self.quick_scan_logo_button.clicked.connect(self.switch_to_quick_scan)

        self.choose_file_button.clicked.connect(self.openFileDialog)
        self.submit_button.clicked.connect(self.switchToLoading)

    def switch_to_dashboard(self):
        self.main_stacked_widget.setCurrentIndex(0)
    
    def switch_to_patients(self):
        self.main_stacked_widget.setCurrentIndex(1)
    
    def switch_to_quick_scan(self):
        self.main_stacked_widget.setCurrentIndex(2)
    
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
                self.image_display_label.setPixmap(pixmap.scaled(self.image_display_label.width(), self.image_display_label.height(), Qt.KeepAspectRatio))
            except Exception as e:
                self.image_display_label.setText(str(e))
                self.currentImagePath = None
    
    def switchToLoading(self):
        if self.currentImagePath is not None:
            self.loadingPage = Loading()
            self.quick_scan_stacked_widget.addWidget(self.loadingPage)  # Add loadingPage to the layout
            self.quick_scan_stacked_widget.setCurrentIndex(self.quick_scan_stacked_widget.count() - 1)      # Switch to loadingPage
            # Use a timer to delay the loading process
            QTimer.singleShot(100, lambda: self.startLoadingProcess())  # 100ms delay
        else:
            self.image_display_label.setText("Please upload an image before submitting.")

    def startLoadingProcess(self):
        diagnosis = processImage(self.currentImagePath)
        self.loadingPage.loadingComplete.connect(lambda: self.switchToReport(diagnosis))

    def switchToReport(self, diagnosis):
        self.reportPage = LesionReport(diagnosis)
        # Remove old LesionReport if it exists
        if self.quick_scan_stacked_widget.count() > 3:
            oldReportPage = self.quick_scan_stacked_widget.widget(3)
            self.quick_scan_stacked_widget.removeWidget(oldReportPage)
            oldReportPage.deleteLater()

        self.quick_scan_stacked_widget.addWidget(self.reportPage)  # Add new reportPage to the layout

        QTimer.singleShot(1000, lambda: self.quick_scan_stacked_widget.setCurrentIndex(self.quick_scan_stacked_widget.count() - 1))  # Delay the switch to show 100% progress
        self.reportPage.backToSubmission.connect(lambda: self.quick_scan_stacked_widget.setCurrentIndex(0))


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
    backToSubmission = pyqtSignal()

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
        self.backToSubmission.emit()