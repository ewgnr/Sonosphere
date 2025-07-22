import numpy as np

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pathlib import Path

class AudioGui(QtWidgets.QWidget):
    
    def __init__(self):
        super().__init__()

        self.setGeometry(50,50,256,256)
        self.setWindowTitle("Clustering")
        