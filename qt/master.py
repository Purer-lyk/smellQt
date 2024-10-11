import sys
import requests
from PyQt5 import QtGui
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QGridLayout, QVBoxLayout, QLabel, QScrollArea
from bs4 import BeautifulSoup
from functools import partial
from plot import GraphProcess
from nnResult import infer
import qdarkstyle
from generate import label_list


class Space(QWidget):
    def __init__(self, mpath):
        super(Space, self).__init__()
        self.div = [[], [], [], []]
        self.model_path = mpath
        self.dataText = QLabel()
        self.area = QScrollArea()
        self.initUI()

    def initUI(self):
        # 创建按钮
        glayout = QGridLayout()
        btnStyleSheet = """
                    QPushButton {
                        color: #F8F8FF;
                        border: 2px solid #555;
                        border-radius: 10px;
                        padding: 5px;
                        background-color: #323278;
                        min-width: 80px;
                        min-height: 10px;
                    }
                    QPushButton:pressed {
                        color: #000000;
                        background-color: #E6E6FA;
                    }
                    QPushButton:hover {
                        border: 2px solid #333;
                    }
                """

        btnr = QPushButton('Result', self)
        btnr.clicked.connect(self.Result)
        btnr.setStyleSheet(btnStyleSheet)
        glayout.addWidget(btnr, 0, 0)

        self.area.setWidget(self.dataText)
        self.area.setWidgetResizable(True)
        self.dataText.setWordWrap(True)
        self.dataText.setAlignment(Qt.AlignCenter)

        self.area.setFixedHeight(40)
        self.area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        glayout.addWidget(self.area, 1, 0)

        self.setLayout(glayout)

    def Result(self):
        print("process data by neural network")
        # todo:调用神经网络部分
        # classIndex = infer(self.model_path, self.div)
        self.dataText.setText(label_list[0])

    def setData(self, input_div):
        self.div = input_div


class box1(QWidget):
    def __init__(self, grapher: GraphProcess, worker: Space, mpath: str):
        super(box1, self).__init__()
        self.div = [[1, 2, 3, 5, 1, 2, 3, 5, 1, 2, 3, 5, 1, 2, 3, 5, 1, 2, 3, 5, 1, 2, 3, 5, 1, 2, 3, 5, 1, 2, 3, 5, 1, 2, 3, 5, 1, 2, 3, 5, 1, 2, 3, 5, 1, 2, 3, 5, 1, 2, 3, 5, 1, 2, 3, 5, 1, 2, 3, 5],
                    [409500, 2, 3, 5, 1, 2, 3, 5, 1, 2, 3, 5, 1, 2, 3, 5, 1, 2, 3, 5, 1, 2, 3, 5, 1, 2, 3, 5, 1, 2, 3, 5, 1, 2, 3, 5, 1, 2, 3, 5, 1, 2, 3, 5, 1, 2, 3, 5, 1, 2, 3, 5, 1, 2, 3, 5, 1, 2, 3, 5], [], []]
        self.grapher = grapher
        self.model_path = mpath
        self.infere = worker
        self.showData = QLabel()
        self.initUI()

    def initUI(self):
        # 创建gui页面布局
        glayout = QGridLayout()

        # 创建按钮
        btnStyleSheet = """
            QPushButton {
                color: #F8F8FF;
                border: 2px solid #555;
                border-radius: 10px;
                padding: 5px;
                background-color: #323278;
                min-width: 80px;
                min-height: 30px;
            }
            QPushButton:pressed {
                color: #000000;
                background-color: #E6E6FA;
            }
            QPushButton:hover {
                border: 2px solid #333;
            }
        """

        btns = QPushButton('Start', self)
        btns.clicked.connect(self.Start)
        btns.setStyleSheet(btnStyleSheet)

        btn1 = QPushButton('Sensor1', self)
        btn1.clicked.connect(partial(self.plotData, 1))
        btn1.setStyleSheet(btnStyleSheet)
        btn2 = QPushButton('Sensor2', self)
        btn2.clicked.connect(partial(self.plotData, 2))
        btn2.setStyleSheet(btnStyleSheet)
        btn3 = QPushButton('Sensor3', self)
        btn3.clicked.connect(partial(self.plotData, 3))
        btn3.setStyleSheet(btnStyleSheet)
        btn4 = QPushButton('Sensor4', self)
        btn4.clicked.connect(partial(self.plotData, 4))
        btn4.setStyleSheet(btnStyleSheet)

        glayout.addWidget(btns, 2, 0, 1, 2)
        glayout.addWidget(btn1, 0, 0)
        glayout.addWidget(btn2, 0, 1)
        glayout.addWidget(btn3, 1, 0)
        glayout.addWidget(btn4, 1, 1)

        self.setLayout(glayout)
        self.resize(500, 600)

    def plotData(self, index):
        print(index)
        getData = [155076, 155425, 155650, 155338, 156342, 156834, 157561, 158391, 159224, 160354, 161646, 163133,
                        164744, 166296, 167988, 169821, 171710, 173896, 176027, 178204, 180745, 183233, 185709, 188565,
                        191415, 194311, 197540, 200053, 204143, 208254, 212051, 215984, 219838, 223793, 228726, 233065,
                        236800, 242232, 247216, 251727, 257412, 262174, 266984, 271537, 276200, 280532, 283758, 288956,
                        292835, 296567, 300459, 303916, 307647, 310991, 313897, 316765, 318440, 321663, 324403, 326498,
                        328457, 330750, 332598, 334384, 336007, 337599, 339038, 340386, 341284, 342024, 342348, 342679,
                        342526, 342713, 343251, 343930, 343884, 345136, 345816, 346575, 347160, 345555, 345555, 345555,
                        345555, 345555, 345555, 345555, 345555, 345555, 345555, 345555, 345555, 345555, 345555, 345555,
                        345555, 345555, 345555, 345555, 345555, 345555, 345555, 345555, 345555, 345555, 345555, 345555,
                        355124, 356083, 356970, 357417, 355756, 375696, 350072, 350476, 351278, 351983, 352549, 350508,
                        335791, 345733, 355432, 365216, 365758, 365779, 357938, 358415, 358927, 365762, 357775, 405686,
                        401210, 407523, 405148, 406514, 409543, 401597, 405742, 405012, 402588, 404544, 402576, 407501
                        ]
        # self.grapher.setData(self.div[index-1])
        self.grapher.setData(getData)

    def Start(self):
        res = requests.post('http://192.168.43.115/Start')
        html = res.text
        soup = BeautifulSoup(html, 'html.parser')
        # print(soup)
        self.div = [[int(i) for i in soup.find('div1').text.split(', ')],
                    [int(i) for i in soup.find('div2').text.split(', ')],
                    [int(i) for i in soup.find('div3').text.split(', ')],
                    [int(i) for i in soup.find('div4').text.split(', ')]]
        print(self.div)
        self.infere.setData(self.div)


class SimpleApp(QWidget):
    def __init__(self, mpath):
        super(SimpleApp, self).__init__()
        self.dragPosition = QPoint(0, 0)
        # 设置gui大小
        self.gwidth = 1500
        self.gheight = 800
        self.ygridSize = 20000
        self.ymaxSize = 200000
        self.yguiSize = int(self.gheight * 0.75)
        self.xguiSize = int(self.gwidth * 0.75)
        self.model_path = mpath
        self.initUI()

    def initUI(self):
        # gui标题
        self.setWindowTitle("...")

        # 创建gui页面布局
        glayout = QGridLayout()

        graph = GraphProcess()
        # div = scroll()
        space1 = Space(self.model_path)
        b1 = box1(graph, space1, self.model_path)
        glayout.addWidget(graph, 0, 0, 2, 1)
        glayout.addWidget(b1, 0, 1, 1, 1)
        glayout.addWidget(space1, 1, 1, 1, 1)

        glayout.setRowStretch(0, 2)
        glayout.setColumnStretch(0, 1.5)  # 设置拉伸因子

        self.setLayout(glayout)
        self.resize(self.gwidth, self.gheight)
        self.setWindowOpacity(0.95)
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())


if __name__ == "__main__":
    model = "./model/2024-09-29_STDSMT_0.pt"
    app = QApplication(sys.argv)
    ex = SimpleApp(model)
    ex.show()
    sys.exit(app.exec())
