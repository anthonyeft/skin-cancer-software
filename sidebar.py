from ui_sidebar import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton

class Sidebar(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle('Demo Application')

        self.icon_name_widget.setHidden(True)

        self.dashboard_button.clicked.connect(self.switch_to_dashboard)
        self.dashboard_logo_button.clicked.connect(self.switch_to_dashboard)

        self.patients_button.clicked.connect(self.switch_to_patients)
        self.patients_logo_button.clicked.connect(self.switch_to_patients)
        
        self.quick_scan_button.clicked.connect(self.switch_to_quick_scan)
        self.quick_scan_logo_button.clicked.connect(self.switch_to_quick_scan)

    def switch_to_dashboard(self):
        self.stackedWidget.setCurrentIndex(0)
    
    def switch_to_patients(self):
        self.stackedWidget.setCurrentIndex(1)
    
    def switch_to_quick_scan(self):
        self.stackedWidget.setCurrentIndex(2)