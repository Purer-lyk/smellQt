from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5 import QtChart
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtGui import QPainter, QPen, QColor, QBrush, QFont
from PyQt5.QtWidgets import QApplication, QWidget, QScrollArea, QGridLayout, QLabel
from functools import partial
import sys
from math import sqrt


class GraphProcess(QWidget):
    def __init__(self):
        super(GraphProcess, self).__init__()
        # self.getData = [155076, 155425, 155650, 155338, 156342, 156834, 157561, 158391, 159224, 160354, 161646, 163133,
        #                 164744, 166296, 167988, 169821, 171710, 173896, 176027, 178204, 180745, 183233, 185709, 188565,
        #                 191415, 194311, 197540, 200053, 204143, 208254, 212051, 215984, 219838, 223793, 228726, 233065,
        #                 236800, 242232, 247216, 251727, 257412, 262174, 266984, 271537, 276200, 280532, 283758, 288956,
        #                 292835, 296567, 300459, 303916, 307647, 310991, 313897, 316765, 318440, 321663, 324403, 326498,
        #                 328457, 330750, 332598, 334384, 336007, 337599, 339038, 340386, 341284, 342024, 342348, 342679,
        #                 342526, 342713, 343251, 343930, 343884, 345136, 345816, 346575, 347160, 345555, 345555, 345555,
        #                 345555, 345555, 345555, 345555, 345555, 345555, 345555, 345555, 345555, 345555, 345555, 345555,
        #                 345555, 345555, 345555, 345555, 345555, 345555, 345555, 345555, 345555, 345555, 345555, 345555,
        #                 355124, 356083, 356970, 357417, 355756, 375696, 350072, 350476, 351278, 351983, 352549, 350508,
        #                 335791, 345733, 355432, 365216, 365758, 365779, 357938, 358415, 358927, 365762, 357775, 405686,
        #                 401210, 407523, 405148, 406514, 409543, 401597, 405742, 405012, 402588, 404544, 402576, 407501
        #                 ]
        self.getData = []
        self.series = QtChart.QLineSeries()
        self.chart = QtChart.QChart()
        self.chartView = QtChart.QChartView(self.chart)
        self.scrollArea = QScrollArea()
        self.dataScroll = QLabel()
        self.initUI()

    def initUI(self):
        layout = QGridLayout()
        axisX = QtChart.QValueAxis()
        axisY = QtChart.QValueAxis()
        axisX.setRange(0, 144)
        axisY.setRange(0, 409500)
        axisX.setLabelsColor(Qt.white)
        axisY.setLabelsColor(Qt.white)
        axisX.setLabelFormat("%d")
        axisY.setLabelFormat("%d")
        self.chart.addSeries(self.series)
        self.chart.addAxis(axisX, QtCore.Qt.AlignBottom)
        self.chart.addAxis(axisY, QtCore.Qt.AlignLeft)
        self.series.attachAxis(axisX)
        self.series.attachAxis(axisY)
        self.series.setName("Digital Voltage")

        titleFont = QFont("Times New Roman", 20, QFont.Bold)
        self.chart.setTitleFont(titleFont)
        self.chart.setTitleBrush(QColor("white"))

        legend = self.chart.legend()
        legend.setFont(QFont("Times New Roman", 15))
        legend.setLabelBrush(QBrush(Qt.white))

        self.chart.setTitle("Gas temperature modulation response")
        self.chart.setAnimationOptions(QtChart.QChart.SeriesAnimations)
        self.chart.setBackgroundBrush(QBrush(QColor(26, 35, 44)))
        self.chartView.setRenderHint(QPainter.Antialiasing)
        layout.addWidget(self.chartView, 0, 0)

        # self.scrollArea.setFixedHeight(50)
        self.scrollArea.setWidget(self.dataScroll)
        self.scrollArea.setWidgetResizable(True)
        layout.addWidget(self.scrollArea, 1, 0)

        layout.setRowStretch(0, 2)

        self.setLayout(layout)
        self.resize(1000, 1000)

    def setData(self, datalist):
        print("update data")
        # if len(datalist) > 1:
        self.series.clear()
        self.getData = datalist
        self.dataScroll.setText(str(datalist))
        for item, idata in enumerate(datalist):
            self.series.append(item, idata)


if __name__ == "__main__":
    # app = QApplication(sys.argv)
    # graph = GraphProcess()
    # graph.show()
    # sys.exit(app.exec())
    a = [1, 2, 2, 3]
    b = [2, 3, 4, 5]
    print(str(b))
    # for index, (i, j) in enumerate(zip(a, b)):
    #     print(index)
    #     print(i)
    #     print(j)
    # print(a[:-1])
    # print(a[1:])
