# General style settings
GENERAL_STYLE = """
    QWidget {
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 30pt;
        color: #333333;
        background-color: #FFFFFF;
    }
"""

# Button style
BUTTON_STYLE = """
    QPushButton {
        background-color: #007BFF;
        border: none;
        color: white;
        padding: 10px 40px;
        text-align: center;
        font-size: 30pt;
        border-radius: 4px;
    }
    QPushButton:hover {
        background-color: #0069D9;
    }
    QPushButton:pressed {
        background-color: #0056B3;
    }
"""

# Label style
LABEL_STYLE = """
    QLabel {
        font-size: 30pt;
        color: #333333;
    }
"""

# Editable text fields (like QLineEdit, QTextEdit)
TEXT_INPUT_STYLE = """
    QLineEdit, QTextEdit {
        border: 1px solid #CCCCCC;
        padding: 4px;
        border-radius: 4px;
    }
    QLineEdit:focus, QTextEdit:focus {
        border-color: #007BFF;
    }
"""

# Main window style
MAIN_WINDOW_STYLE = """
    QMainWindow {
        background-color: #F8F9FA;
    }
"""

# Combining all styles
STYLESHEET = GENERAL_STYLE + BUTTON_STYLE + LABEL_STYLE + TEXT_INPUT_STYLE + MAIN_WINDOW_STYLE
