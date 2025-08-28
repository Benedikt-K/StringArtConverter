from __future__ import annotations
import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon
from StringArtConverter.UI.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("String Art Converter")

    app.setWindowIcon(QIcon("icon.png"))

    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()