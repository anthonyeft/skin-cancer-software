from PyQt5.QtWidgets import QApplication
import sys
from ui_handler import mainApplication
import warnings

warnings.filterwarnings("ignore")

### Main Function ###

app = QApplication(sys.argv)
window = mainApplication()
window.showFullScreen()

app.exec_()