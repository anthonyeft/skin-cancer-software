from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton
import sys
from sidebar import Sidebar

app = QApplication(sys.argv)
window = Sidebar()
window.show()
app.exec_()