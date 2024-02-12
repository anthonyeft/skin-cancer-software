from ui import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QObject
from utils.process_image import processImage

class DiagnosisWorker(QObject):
    finished = pyqtSignal(str)

    def __init__(self, imagePath):
        super().__init__()
        self.imagePath = imagePath

    def run(self):
        # Perform the image diagnosis
        diagnosis = processImage(self.imagePath)
        self.finished.emit(diagnosis)  # Emit the diagnosis

class mainApplication(QMainWindow, Ui_MainWindow):
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
            self.quick_scan_stacked_widget.setCurrentIndex(1)

            # Setup the progress bar
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.updateProgressBar)
            self.timer.start(25)
            self.progress = 0

            # Setup and start the diagnosis thread
            self.thread = QThread()
            self.diagnosis_worker = DiagnosisWorker(self.currentImagePath)
            self.diagnosis_worker.moveToThread(self.thread)
            self.diagnosis_worker.finished.connect(self.diagnosisComplete)
            self.thread.started.connect(self.diagnosis_worker.run)
            self.thread.start()

        else:
            self.image_display_label.setText("Please upload an image before submitting.")
        
    def diagnosisComplete(self, diagnosis):
        #self.diagnosis = diagnosis
        self.switchToReport(diagnosis)
        if self.progress < 100:
            self.progress = 100
            self.progress_bar.setValue(self.progress)
        self.thread.quit()
        self.thread.wait()

    def updateProgressBar(self):
        self.progress += 1  # Increment the progress
        self.progress_bar.setValue(self.progress)

        if self.progress >= 100:
            self.timer.stop()

    def switchToReport(self, diagnosis):
        self.diagnosis_label.setText(f"Diagnosis: {diagnosis}")
        pixmap = QPixmap(self.currentImagePath)
        scaled_pixmap = pixmap.scaled(self.processed_image_label.width(), self.processed_image_label.height(), Qt.KeepAspectRatio)
        self.processed_image_label.setPixmap(scaled_pixmap)
        self.processed_image_label.setFixedWidth(scaled_pixmap.width())
        self.processed_image_label.setAlignment(Qt.AlignCenter)
        QTimer.singleShot(1000, lambda: self.quick_scan_stacked_widget.setCurrentIndex(2))