from PyQt5.QtWidgets import QApplication
import sys
from ui_handler import Sidebar

### Main Function ###

app = QApplication(sys.argv)
window = Sidebar()
window.show()
app.exec_()