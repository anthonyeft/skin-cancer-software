# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/main.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1920, 1080)
        MainWindow.setStyleSheet("background-color: rgb(248, 249, 250);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setSpacing(0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.icon_only_sidebar_widget = QtWidgets.QWidget(self.centralwidget)
        self.icon_only_sidebar_widget.setStyleSheet("QWidget{\n"
"    background-color: rgb(0, 123, 255);\n"
"}\n"
"QPushButton{\n"
"    border:none;\n"
"    border-radius:10px;\n"
"}\n"
"QPushButton:checked{\n"
"    background-color:#F8F9FA;\n"
"    color: rgb(0, 123, 255);\n"
"    font-weight:bold;\n"
"}")
        self.icon_only_sidebar_widget.setObjectName("icon_only_sidebar_widget")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.icon_only_sidebar_widget)
        self.verticalLayout_3.setContentsMargins(10, -1, 10, -1)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.empty_spacer_label = QtWidgets.QLabel(self.icon_only_sidebar_widget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.empty_spacer_label.setFont(font)
        self.empty_spacer_label.setText("")
        self.empty_spacer_label.setAlignment(QtCore.Qt.AlignCenter)
        self.empty_spacer_label.setObjectName("empty_spacer_label")
        self.verticalLayout_3.addWidget(self.empty_spacer_label)
        self.icon_only_widget = QtWidgets.QVBoxLayout()
        self.icon_only_widget.setSpacing(20)
        self.icon_only_widget.setObjectName("icon_only_widget")
        self.dashboard_logo_button = QtWidgets.QPushButton(self.icon_only_sidebar_widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.dashboard_logo_button.sizePolicy().hasHeightForWidth())
        self.dashboard_logo_button.setSizePolicy(sizePolicy)
        self.dashboard_logo_button.setMinimumSize(QtCore.QSize(100, 100))
        self.dashboard_logo_button.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/icons/dashboard_white.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon.addPixmap(QtGui.QPixmap(":/icons/icons/dashboard_blue.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.dashboard_logo_button.setIcon(icon)
        self.dashboard_logo_button.setIconSize(QtCore.QSize(60, 60))
        self.dashboard_logo_button.setCheckable(True)
        self.dashboard_logo_button.setAutoExclusive(True)
        self.dashboard_logo_button.setObjectName("dashboard_logo_button")
        self.icon_only_widget.addWidget(self.dashboard_logo_button)
        self.patients_logo_button = QtWidgets.QPushButton(self.icon_only_sidebar_widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.patients_logo_button.sizePolicy().hasHeightForWidth())
        self.patients_logo_button.setSizePolicy(sizePolicy)
        self.patients_logo_button.setMinimumSize(QtCore.QSize(100, 100))
        self.patients_logo_button.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icons/icons/person_white.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon1.addPixmap(QtGui.QPixmap(":/icons/icons/person_blue.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.patients_logo_button.setIcon(icon1)
        self.patients_logo_button.setIconSize(QtCore.QSize(60, 60))
        self.patients_logo_button.setCheckable(True)
        self.patients_logo_button.setAutoExclusive(True)
        self.patients_logo_button.setObjectName("patients_logo_button")
        self.icon_only_widget.addWidget(self.patients_logo_button)
        self.quick_scan_logo_button = QtWidgets.QPushButton(self.icon_only_sidebar_widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.quick_scan_logo_button.sizePolicy().hasHeightForWidth())
        self.quick_scan_logo_button.setSizePolicy(sizePolicy)
        self.quick_scan_logo_button.setMinimumSize(QtCore.QSize(100, 100))
        self.quick_scan_logo_button.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/icons/icons/scan_white.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon2.addPixmap(QtGui.QPixmap(":/icons/icons/scan_blue.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.quick_scan_logo_button.setIcon(icon2)
        self.quick_scan_logo_button.setIconSize(QtCore.QSize(60, 60))
        self.quick_scan_logo_button.setCheckable(True)
        self.quick_scan_logo_button.setAutoExclusive(True)
        self.quick_scan_logo_button.setObjectName("quick_scan_logo_button")
        self.icon_only_widget.addWidget(self.quick_scan_logo_button)
        self.verticalLayout_3.addLayout(self.icon_only_widget)
        spacerItem = QtWidgets.QSpacerItem(20, 356, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem)
        self.close_logo_button = QtWidgets.QPushButton(self.icon_only_sidebar_widget)
        self.close_logo_button.setMinimumSize(QtCore.QSize(100, 100))
        self.close_logo_button.setText("")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/icons/icons/close_white.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.close_logo_button.setIcon(icon3)
        self.close_logo_button.setIconSize(QtCore.QSize(60, 60))
        self.close_logo_button.setCheckable(True)
        self.close_logo_button.setAutoExclusive(True)
        self.close_logo_button.setObjectName("close_logo_button")
        self.verticalLayout_3.addWidget(self.close_logo_button)
        self.gridLayout_2.addWidget(self.icon_only_sidebar_widget, 0, 0, 1, 1)
        self.icon_name_sidebar_widget = QtWidgets.QWidget(self.centralwidget)
        self.icon_name_sidebar_widget.setStyleSheet("QWidget{\n"
"    background-color: rgb(0, 123, 255);\n"
"    color:white;\n"
"}\n"
"QPushButton{\n"
"    color:white;\n"
"    border:none;\n"
"    text-align:left;\n"
"    border-top-left-radius:10px;\n"
"    border-bottom-left-radius:10px;\n"
"}\n"
"QPushButton:checked{\n"
"    background-color:#F8F9FA;\n"
"    color: rgb(0, 123, 255);\n"
"    font-weight:bold;\n"
"}")
        self.icon_name_sidebar_widget.setObjectName("icon_name_sidebar_widget")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.icon_name_sidebar_widget)
        self.verticalLayout_4.setContentsMargins(30, -1, 0, -1)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.sidebar_label = QtWidgets.QLabel(self.icon_name_sidebar_widget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.sidebar_label.setFont(font)
        self.sidebar_label.setAlignment(QtCore.Qt.AlignCenter)
        self.sidebar_label.setObjectName("sidebar_label")
        self.verticalLayout_4.addWidget(self.sidebar_label)
        self.icon_name_only_widget = QtWidgets.QVBoxLayout()
        self.icon_name_only_widget.setSpacing(20)
        self.icon_name_only_widget.setObjectName("icon_name_only_widget")
        self.dashboard_button = QtWidgets.QPushButton(self.icon_name_sidebar_widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.dashboard_button.sizePolicy().hasHeightForWidth())
        self.dashboard_button.setSizePolicy(sizePolicy)
        self.dashboard_button.setMinimumSize(QtCore.QSize(280, 100))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.dashboard_button.setFont(font)
        self.dashboard_button.setIcon(icon)
        self.dashboard_button.setIconSize(QtCore.QSize(60, 60))
        self.dashboard_button.setCheckable(True)
        self.dashboard_button.setAutoExclusive(True)
        self.dashboard_button.setObjectName("dashboard_button")
        self.icon_name_only_widget.addWidget(self.dashboard_button)
        self.patients_button = QtWidgets.QPushButton(self.icon_name_sidebar_widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.patients_button.sizePolicy().hasHeightForWidth())
        self.patients_button.setSizePolicy(sizePolicy)
        self.patients_button.setMinimumSize(QtCore.QSize(280, 100))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.patients_button.setFont(font)
        self.patients_button.setIcon(icon1)
        self.patients_button.setIconSize(QtCore.QSize(60, 60))
        self.patients_button.setCheckable(True)
        self.patients_button.setAutoExclusive(True)
        self.patients_button.setObjectName("patients_button")
        self.icon_name_only_widget.addWidget(self.patients_button)
        self.quick_scan_button = QtWidgets.QPushButton(self.icon_name_sidebar_widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.quick_scan_button.sizePolicy().hasHeightForWidth())
        self.quick_scan_button.setSizePolicy(sizePolicy)
        self.quick_scan_button.setMinimumSize(QtCore.QSize(280, 100))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.quick_scan_button.setFont(font)
        self.quick_scan_button.setIcon(icon2)
        self.quick_scan_button.setIconSize(QtCore.QSize(60, 60))
        self.quick_scan_button.setCheckable(True)
        self.quick_scan_button.setAutoExclusive(True)
        self.quick_scan_button.setObjectName("quick_scan_button")
        self.icon_name_only_widget.addWidget(self.quick_scan_button)
        self.verticalLayout_4.addLayout(self.icon_name_only_widget)
        spacerItem1 = QtWidgets.QSpacerItem(20, 356, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_4.addItem(spacerItem1)
        self.close_button = QtWidgets.QPushButton(self.icon_name_sidebar_widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.close_button.sizePolicy().hasHeightForWidth())
        self.close_button.setSizePolicy(sizePolicy)
        self.close_button.setMinimumSize(QtCore.QSize(200, 100))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.close_button.setFont(font)
        self.close_button.setIcon(icon3)
        self.close_button.setIconSize(QtCore.QSize(60, 60))
        self.close_button.setCheckable(True)
        self.close_button.setAutoExclusive(True)
        self.close_button.setObjectName("close_button")
        self.verticalLayout_4.addWidget(self.close_button)
        self.gridLayout_2.addWidget(self.icon_name_sidebar_widget, 0, 1, 1, 1)
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setObjectName("widget")
        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setObjectName("gridLayout")
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem2, 0, 1, 1, 1)
        self.menu_button = QtWidgets.QPushButton(self.widget)
        self.menu_button.setMinimumSize(QtCore.QSize(60, 60))
        self.menu_button.setStyleSheet("border:none;")
        self.menu_button.setText("")
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/icons/icons/menu_black.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.menu_button.setIcon(icon4)
        self.menu_button.setIconSize(QtCore.QSize(60, 60))
        self.menu_button.setCheckable(True)
        self.menu_button.setObjectName("menu_button")
        self.gridLayout.addWidget(self.menu_button, 0, 0, 1, 1)
        self.main_stacked_widget = QtWidgets.QStackedWidget(self.widget)
        self.main_stacked_widget.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"")
        self.main_stacked_widget.setObjectName("main_stacked_widget")
        self.dashboard_page = QtWidgets.QWidget()
        self.dashboard_page.setObjectName("dashboard_page")
        self.dashboard_label = QtWidgets.QLabel(self.dashboard_page)
        self.dashboard_label.setGeometry(QtCore.QRect(320, 220, 191, 61))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.dashboard_label.setFont(font)
        self.dashboard_label.setObjectName("dashboard_label")
        self.main_stacked_widget.addWidget(self.dashboard_page)
        self.patients_page = QtWidgets.QWidget()
        self.patients_page.setObjectName("patients_page")
        self.patient_label = QtWidgets.QLabel(self.patients_page)
        self.patient_label.setGeometry(QtCore.QRect(370, 200, 161, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.patient_label.setFont(font)
        self.patient_label.setObjectName("patient_label")
        self.main_stacked_widget.addWidget(self.patients_page)
        self.quick_scan_page = QtWidgets.QWidget()
        self.quick_scan_page.setStyleSheet("QPushButton {\n"
"    background-color: #007BFF;\n"
"    border: none;\n"
"    color: white;\n"
"    padding: 6px 24px;\n"
"    text-align: center;\n"
"    font-size: 10pt;\n"
"    border-radius: 8px;\n"
"}\n"
"QPushButton:hover {\n"
"    background-color: #0069D9;\n"
"}\n"
"QPushButton:pressed {\n"
"    background-color: #0056B3;\n"
"}\n"
"")
        self.quick_scan_page.setObjectName("quick_scan_page")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.quick_scan_page)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.quick_scan_stacked_widget = QtWidgets.QStackedWidget(self.quick_scan_page)
        self.quick_scan_stacked_widget.setObjectName("quick_scan_stacked_widget")
        self.submission_page = QtWidgets.QWidget()
        self.submission_page.setObjectName("submission_page")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.submission_page)
        self.verticalLayout_6.setContentsMargins(0, 10, 0, 0)
        self.verticalLayout_6.setSpacing(10)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.quick_scan_label = QtWidgets.QLabel(self.submission_page)
        self.quick_scan_label.setMinimumSize(QtCore.QSize(0, 100))
        self.quick_scan_label.setMaximumSize(QtCore.QSize(1000000, 70))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.quick_scan_label.setFont(font)
        self.quick_scan_label.setStyleSheet("font-size: 20pt;\n"
"color: #333333;\n"
"font-weight:bold;")
        self.quick_scan_label.setAlignment(QtCore.Qt.AlignCenter)
        self.quick_scan_label.setObjectName("quick_scan_label")
        self.verticalLayout_6.addWidget(self.quick_scan_label)
        spacerItem3 = QtWidgets.QSpacerItem(20, 80, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_6.addItem(spacerItem3)
        self.image_h_frame = QtWidgets.QFrame(self.submission_page)
        self.image_h_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.image_h_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.image_h_frame.setObjectName("image_h_frame")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.image_h_frame)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem4)
        self.image_display_label = QtWidgets.QLabel(self.image_h_frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.image_display_label.sizePolicy().hasHeightForWidth())
        self.image_display_label.setSizePolicy(sizePolicy)
        self.image_display_label.setMinimumSize(QtCore.QSize(600, 450))
        self.image_display_label.setMaximumSize(QtCore.QSize(1200, 900))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.image_display_label.setFont(font)
        self.image_display_label.setStyleSheet("border: 2px dashed #7A909E;\n"
"color: #3E5463;\n"
"background-color: #F2F5F7;\n"
"font-weight: bold;\n"
"font-size: 12pt;")
        self.image_display_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_display_label.setObjectName("image_display_label")
        self.horizontalLayout_2.addWidget(self.image_display_label)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem5)
        self.verticalLayout_6.addWidget(self.image_h_frame)
        self.button_h_frame = QtWidgets.QFrame(self.submission_page)
        self.button_h_frame.setMinimumSize(QtCore.QSize(0, 80))
        self.button_h_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.button_h_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.button_h_frame.setObjectName("button_h_frame")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.button_h_frame)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(50)
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem6)
        self.choose_file_button = QtWidgets.QPushButton(self.button_h_frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.choose_file_button.sizePolicy().hasHeightForWidth())
        self.choose_file_button.setSizePolicy(sizePolicy)
        self.choose_file_button.setMinimumSize(QtCore.QSize(250, 50))
        self.choose_file_button.setMaximumSize(QtCore.QSize(200, 40))
        self.choose_file_button.setObjectName("choose_file_button")
        self.horizontalLayout.addWidget(self.choose_file_button)
        self.submit_button = QtWidgets.QPushButton(self.button_h_frame)
        self.submit_button.setMinimumSize(QtCore.QSize(200, 50))
        self.submit_button.setMaximumSize(QtCore.QSize(200, 40))
        self.submit_button.setObjectName("submit_button")
        self.horizontalLayout.addWidget(self.submit_button)
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem7)
        self.verticalLayout_6.addWidget(self.button_h_frame)
        spacerItem8 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_6.addItem(spacerItem8)
        self.quick_scan_stacked_widget.addWidget(self.submission_page)
        self.verticalLayout.addWidget(self.quick_scan_stacked_widget)
        self.main_stacked_widget.addWidget(self.quick_scan_page)
        self.gridLayout.addWidget(self.main_stacked_widget, 1, 0, 1, 2)
        self.gridLayout_2.addWidget(self.widget, 0, 2, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.menu_button.toggled['bool'].connect(self.icon_only_sidebar_widget.setHidden) # type: ignore
        self.menu_button.toggled['bool'].connect(self.icon_name_sidebar_widget.setVisible) # type: ignore
        self.quick_scan_logo_button.toggled['bool'].connect(self.quick_scan_button.setChecked) # type: ignore
        self.patients_logo_button.toggled['bool'].connect(self.patients_button.setChecked) # type: ignore
        self.dashboard_logo_button.toggled['bool'].connect(self.dashboard_button.setChecked) # type: ignore
        self.dashboard_button.toggled['bool'].connect(self.dashboard_logo_button.setChecked) # type: ignore
        self.patients_button.toggled['bool'].connect(self.patients_logo_button.setChecked) # type: ignore
        self.quick_scan_button.toggled['bool'].connect(self.quick_scan_logo_button.setChecked) # type: ignore
        self.close_logo_button.toggled['bool'].connect(MainWindow.close) # type: ignore
        self.close_button.toggled['bool'].connect(MainWindow.close) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.sidebar_label.setText(_translate("MainWindow", "Sidebar"))
        self.dashboard_button.setText(_translate("MainWindow", " Dashboard"))
        self.patients_button.setText(_translate("MainWindow", " Patients"))
        self.quick_scan_button.setText(_translate("MainWindow", " Quick Scan"))
        self.close_button.setText(_translate("MainWindow", "Close"))
        self.dashboard_label.setText(_translate("MainWindow", "dashboard"))
        self.patient_label.setText(_translate("MainWindow", "patients"))
        self.quick_scan_label.setText(_translate("MainWindow", "Quick Scan"))
        self.image_display_label.setText(_translate("MainWindow", "Image will be displayed here"))
        self.choose_file_button.setText(_translate("MainWindow", "Choose File"))
        self.submit_button.setText(_translate("MainWindow", "Submit"))
import resource_rc


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
