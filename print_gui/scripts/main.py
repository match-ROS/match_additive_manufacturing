#! /usr/bin/env python3
import sys
from PyQt5.QtWidgets import QApplication
from gui_layout import ROSGui

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ROSGui()
    window.show()
    sys.exit(app.exec_())
