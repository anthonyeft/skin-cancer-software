from PyQt5.QtWidgets import QApplication
import sys
from ui_handler import mainApplication

### Main Function ###

app = QApplication(sys.argv)
window = mainApplication()
window.show()
app.exec_()