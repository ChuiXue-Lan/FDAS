#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/29  20:26
# @Author  : 菠萝吹雪
# @Software: PyCharm
# @Describe: 
# -*- encoding:utf-8 -*-
import sys

from PySide6.QtWidgets import QMainWindow, QApplication

from ui.home import Ui_MainWindow


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        # basic interface
        # 创建页面
        self.setupUi(self)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    Home = MainWindow()
    Home.show()
    sys.exit(app.exec())