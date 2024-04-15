#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/17  14:35
# @Author  : 菠萝吹雪
# @Software: PyCharm
# @Describe: 
# -*- encoding:utf-8 -*-
import sys
from PySide6.QtWidgets import QApplication, QWidget
from ui.utils.tts_dialog import Ui_TTSDialog


class TTSWindow(QWidget, Ui_TTSDialog):
    def __init__(self):
        super(TTSWindow, self).__init__()
        self.setupUi(self)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TTSWindow()
    window.show()
    sys.exit(app.exec())
