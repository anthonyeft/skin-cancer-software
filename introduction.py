from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtCore import Qt

class IntroductionPage(QWidget):
    def __init__(self, switch_callback):
        super().__init__()
        self.switch_callback = switch_callback
        self.initUI()

    def initUI(self):
        introLabel = QLabel("Welcome to the Skin Lesion Diagnosis App.\nPlease click 'Start' to begin.")
        introLabel.setAlignment(Qt.AlignCenter)
        startButton = QPushButton("Start")
        startButton.clicked.connect(self.switchToMain)

        layout = QVBoxLayout()
        layout.addWidget(introLabel)
        layout.addWidget(startButton)
        self.setLayout(layout)

    def switchToMain(self):
        self.switch_callback("main")