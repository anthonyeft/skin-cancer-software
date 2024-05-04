from ui import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QObject

from process_image import processImage, processLesionEvolution
from utils import convertArrayToPixmap, ABCWidget

class DiagnosisWorker(QObject):
    finished = pyqtSignal(str, object, object, object, object, float, float, float, str)

    def __init__(self, imagePath):
        super().__init__()
        self.imagePath = imagePath

    def run(self):
        # Perform the image processing algorithms
        diagnosis, color_constancy_image, contour_image, cam_image, colors_image, asymmetry_score, border_score, color_score, traits = processImage(self.imagePath)
        self.finished.emit(diagnosis, color_constancy_image, contour_image, cam_image, colors_image, asymmetry_score, border_score, color_score, traits)

class mainApplication(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle('Demo Application')

        self.icon_name_sidebar_widget.setHidden(True)

        # Sidebar buttons
        self.evolution_button.clicked.connect(self.switch_to_evolution)
        self.evolution_logo_button.clicked.connect(self.switch_to_evolution)
       
        self.quick_scan_button.clicked.connect(self.switch_to_quick_scan)
        self.quick_scan_logo_button.clicked.connect(self.switch_to_quick_scan)

        # Evolution submission page buttons
        self.upload_left.clicked.connect(self.openFileDialogLeft)
        self.upload_right.clicked.connect(self.openFileDialogRight)
        self.submit_tracking_button.clicked.connect(self.switch_to_evolution_report)
        self.evolution_tracking_results_back_button.clicked.connect(self.switch_to_evolution_submission)

        # Quick scan submission page buttons
        self.choose_file_button.clicked.connect(self.openFileDialog)
        self.submit_button.clicked.connect(self.switchToLoading)

        # Report page buttons
        self.original_image_next_button.clicked.connect(self.switchToColorConstancyImage)

        self.colour_constancy_image_previous_button.clicked.connect(self.switchToOriginalImage)
        self.colour_constancy_image_next_button.clicked.connect(self.switchToSegmentedImage)

        self.segmented_image_previous_button.clicked.connect(self.switchToColorConstancyImage)
        self.segmented_image_next_button.clicked.connect(self.switchToCamImage)

        self.cam_image_previous_button.clicked.connect(self.switchToSegmentedImage)
        self.cam_image_next_button.clicked.connect(self.switchToColorsImage)

        self.colors_image_previous_button.clicked.connect(self.switchToCamImage)
        self.colors_image_next_button.clicked.connect(self.switchToOriginalImage)

        self.back_to_scan_button.clicked.connect(self.switch_to_quick_scan)

        # Report page ABC metrics
        self.ABCWidgetA = ABCWidget("A", self.A_placeholder_widget)
        self.A_placeholder_widget.layout().addWidget(self.ABCWidgetA)

        self.ABCWidgetB = ABCWidget("B", self.B_placeholder_widget)
        self.B_placeholder_widget.layout().addWidget(self.ABCWidgetB)

        self.ABCWidgetC = ABCWidget("C", self.C_placeholder_widget)
        self.C_placeholder_widget.layout().addWidget(self.ABCWidgetC)

        # Empty image paths
        self.currentImagePath = None
        self.leftImagePath = None
        self.rightImagePath = None
    
    def switch_to_quick_scan(self):
        self.image_display_label.clear()
        self.image_display_label.setText("Image will be displayed here")
        self.currentImagePath = None
        self.main_stacked_widget.setCurrentIndex(0)
        self.quick_scan_stacked_widget.setCurrentIndex(0)
        self.report_left_panel_stacked_widget.setCurrentIndex(0)
    
    def switch_to_evolution(self):
        self.main_stacked_widget.setCurrentIndex(1)

    def switch_to_evolution_report(self):
        if self.leftImagePath is not None and self.rightImagePath is not None:
            self.evolution_mask = processLesionEvolution(self.leftImagePath, self.rightImagePath)
            self.segmentation_comparison_image_label.setPixmap(convertArrayToPixmap(self.evolution_mask))
            self.evolution_tracking_stacked_widget.setCurrentIndex(1)
    
    def switch_to_evolution_submission(self):
        self.evolution_tracking_stacked_widget.setCurrentIndex(0)
    
    def openFileDialog(self):
        options = QFileDialog.Options()
        initialDir = "C:\\test_images"
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image", initialDir, "Image Files (*.jpg *.png)", options=options)
        if fileName:
            try:
                if not fileName.lower().endswith(('.png', '.jpg', '.jpeg')):
                    raise ValueError("File format not supported. Please select a JPG or PNG file.")
                self.currentImagePath = fileName  # Set the current image path
                pixmap = QPixmap(fileName)
                self.image_display_label.setPixmap(pixmap.scaled(self.image_display_label.width(), self.image_display_label.height()))
            except Exception as e:
                self.image_display_label.setText(str(e))
                self.currentImagePath = None
    
    def openFileDialogLeft(self):
        options = QFileDialog.Options()
        initialDir = "C:\\test_images"
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image", initialDir, "Image Files (*.jpg *.png)", options=options)
        if fileName:
            try:
                if not fileName.lower().endswith(('.png', '.jpg', '.jpeg')):
                    raise ValueError("File format not supported. Please select a JPG or PNG file.")
                self.leftImagePath = fileName  # Set the current image path
                pixmap = QPixmap(fileName)
                self.left_image_label.setPixmap(pixmap.scaled(self.image_display_label.width(), self.image_display_label.height()))
            except Exception as e:
                self.left_image_label.setText(str(e))
                self.leftImagePath = None
    
    def openFileDialogRight(self):
        options = QFileDialog.Options()
        initialDir = "C:\\test_images"
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image", initialDir, "Image Files (*.jpg *.png)", options=options)
        if fileName:
            try:
                if not fileName.lower().endswith(('.png', '.jpg', '.jpeg')):
                    raise ValueError("File format not supported. Please select a JPG or PNG file.")
                self.rightImagePath = fileName  # Set the current image path
                pixmap = QPixmap(fileName)
                self.right_image_label.setPixmap(pixmap.scaled(self.image_display_label.width(), self.image_display_label.height()))
            except Exception as e:
                self.right_image_label.setText(str(e))
                self.rightImagePath = None
    
    def switchToLoading(self):
        if self.currentImagePath is not None:
            self.progress = 0
            self.quick_scan_stacked_widget.setCurrentIndex(1)
            # Setup the progress bar
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.updateProgressBar)
            self.timer.start(180)

            # Setup and start the diagnosis thread
            self.thread = QThread()
            self.diagnosis_worker = DiagnosisWorker(self.currentImagePath)
            self.diagnosis_worker.moveToThread(self.thread)
            self.diagnosis_worker.finished.connect(self.diagnosisComplete)
            self.thread.started.connect(self.diagnosis_worker.run)
            self.thread.start()

        else:
            self.image_display_label.setText("Please upload an image before submitting.")
        
    def diagnosisComplete(self, diagnosis, color_constancy_image, contour_image, cam_image, colors_image, asymmetry_score, border_score, color_score, traits):
        self.switchToReport(diagnosis, traits, color_constancy_image, contour_image, cam_image, colors_image, asymmetry_score, border_score, color_score)
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

    def switchToReport(
    self,
    diagnosis,
    traits,
    color_constancy_image,
    contour_image,
    cam_image,
    colors_image,
    asymmetry_score,
    border_score,
    color_score,
    ):
        self.diagnosis_label.setText(f"Predicted diagnosis:\n{diagnosis}")
        self.cancerous_traits_label.setText(f"Cancerous Traits:\n{traits}")

        # Display the original image
        original_image_pixmap = QPixmap(self.currentImagePath)
        original_image_pixmap_scaled = original_image_pixmap.scaled(self.original_image_label.width(), self.original_image_label.height())
        self.original_image_label.setPixmap(original_image_pixmap_scaled)
        self.original_image_label.setFixedWidth(original_image_pixmap_scaled.width())

        # Display the color constancy image
        color_constancy_pixmap = convertArrayToPixmap(color_constancy_image)
        color_constancy_pixmap_scaled = color_constancy_pixmap.scaled(self.colour_constancy_image_label.width(), self.colour_constancy_image_label.height(), Qt.KeepAspectRatio)
        self.colour_constancy_image_label.setPixmap(color_constancy_pixmap_scaled)
        self.colour_constancy_image_label.setFixedWidth(color_constancy_pixmap_scaled.width())

        # Display the contour image
        contour_pixmap = convertArrayToPixmap(contour_image)
        contour_pixmap_scaled = contour_pixmap.scaled(self.segmented_image_label.width(), self.segmented_image_label.height(), Qt.KeepAspectRatio)
        self.segmented_image_label.setPixmap(contour_pixmap_scaled)
        self.segmented_image_label.setFixedWidth(contour_pixmap_scaled.width())

        # Display the cam image
        cam_pixmap = convertArrayToPixmap(cam_image)
        cam_pixmap_scaled = cam_pixmap.scaled(self.cam_image_label.width(), self.cam_image_label.height(), Qt.KeepAspectRatio)
        self.cam_image_label.setPixmap(cam_pixmap_scaled)
        self.cam_image_label.setFixedWidth(cam_pixmap_scaled.width())

        # Display the colors image
        colors_pixmap = convertArrayToPixmap(colors_image)
        colors_pixmap_scaled = colors_pixmap.scaled(self.colors_image_label.width(), self.colors_image_label.height(), Qt.KeepAspectRatio)
        self.colors_image_label.setPixmap(colors_pixmap_scaled)
        self.colors_image_label.setFixedWidth(colors_pixmap_scaled.width())

        QTimer.singleShot(500, lambda: self.quick_scan_stacked_widget.setCurrentIndex(2))

        QTimer.singleShot(1300, lambda: self.animateNeedles(asymmetry_score, border_score, color_score))
    
    def animateNeedles(self, asymmetry_score, border_score, color_score):
        self.ABCWidgetA.animateNeedle(asymmetry_score)
        self.ABCWidgetB.animateNeedle(border_score)
        self.ABCWidgetC.animateNeedle(color_score)

    def switchToOriginalImage(self):
        self.report_left_panel_stacked_widget.setCurrentIndex(0)
    
    def switchToColorConstancyImage(self):
        self.report_left_panel_stacked_widget.setCurrentIndex(1)
    
    def switchToSegmentedImage(self):
        self.report_left_panel_stacked_widget.setCurrentIndex(2)
    
    def switchToCamImage(self):
        self.report_left_panel_stacked_widget.setCurrentIndex(3)

    def switchToColorsImage(self):
        self.report_left_panel_stacked_widget.setCurrentIndex(4)