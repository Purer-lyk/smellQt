import sys
from PyQt5.QtCore import Qt, QPoint, QRect
from PyQt5 import QtGui
from PyQt5.QtGui import QPalette
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QToolButton, QTableWidget


class SimpleApp(QWidget):
    def __init__(self):
        super(SimpleApp, self).__init__()
        self.sec_window = second_window()
        self.dragPosition = QPoint(0, 0)
        self.label = QLabel("welcome")
        self.table = QTableWidget(self)
        self.initUI()

    def initUI(self):
        # gui标题
        self.setWindowTitle("...")
        # self.setGeometry(300, 300, 250, 150)

        # table初始化
        self.table.setColumnCount(3)
        self.table.setRowCount(5)

        # 创建gui页面布局
        layout = QVBoxLayout()

        # 创建标签
        self.label.setFixedSize(20, 30)
        self.label.setAutoFillBackground(True)
        self.label.setPalette(QPalette(Qt.red))
        layout.addWidget(self.label)

        # 创建按钮
        btn = QPushButton('click me please', self)
        btn.clicked.connect(self.on_click)

        btn1 = QPushButton('skip to next lay', self)
        btn1.clicked.connect(self.second_window)

        layout.addWidget(btn)
        layout.addWidget(btn1)

        # 设置布局
        self.setLayout(layout)

        # 设置gui大小
        self.resize(1000, 800)

    def on_click(self):
        print("start")

    def second_window(self):
        self.sec_window.show()
        # self.close()

    def mousePressEvent(self, a0: QtGui.QMouseEvent) -> None:
        if a0.button() == Qt.LeftButton:
            self.dragPosition = a0.globalPos() - \
                                self.label.frameGeometry().topLeft()
            print("down")

    def mouseMoveEvent(self, a0: QtGui.QMouseEvent) -> None:
        if a0.buttons() == Qt.LeftButton:
            new_x = a0.globalX() - self.dragPosition.x()
            new_y = a0.globalY() - self.dragPosition.y()
            print(new_y)
            print(new_x)

            # print(self.frameGeometry().size().height())
            # print(self.frameGeometry().size().width())
            limit_area = QRect(0, 0, self.width(), self.height())
            # new_x = max(limit_area.x(),
            #             min(new_x, limit_area.x() + limit_area.width() - self.width()))
            # new_y = max(limit_area.y(),
            #             min(new_y, limit_area.y() + limit_area.height() - self.height()))

            new_x = min(max(limit_area.x(), new_x), limit_area.x() + limit_area.width() - self.label.width())
            new_y = min(max(limit_area.y(), new_y), limit_area.y() + limit_area.height() - self.label.height())

            self.label.move(new_x, new_y)

    def mouseReleaseEvent(self, a0: QtGui.QMouseEvent) -> None:
        pass


class second_window(QWidget):
    def __init__(self):
        super(second_window, self).__init__()
        self.setWindowTitle('sec_window')
        self.setGeometry(0, 0, 1000, 800)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = SimpleApp()
    ex.show()
    sys.exit(app.exec())
