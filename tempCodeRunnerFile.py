import sys
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)

    # Load stylesheet
    with open("D:/HaoHaoHao/Hoc Hanh/HK7/Iot/Đồ án/Đồ án cuối kì/object_recognition_app/ui/style.qss", "r") as f:
        app.setStyleSheet(f.read())

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()