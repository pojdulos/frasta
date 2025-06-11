
from pyqtgraph.Qt import QtWidgets, QtCore
import pyqtgraph as pg
import numpy as np

class PixelSnapViewBox(pg.ViewBox):
    def __init__(self):
        super().__init__()    
    
    # def wheelEvent(self, ev, axis=None):
    #     mods = ev.modifiers()

    #     step = 1 if not (mods & QtCore.Qt.ControlModifier) else max(1, int(abs(ev.delta())))
    #     delta = int(ev.delta() / 120) * step
    #     if delta == 0:
    #         return

    #     x_range, y_range = self.viewRange()
    #     x_min, x_max = int(round(x_range[0])), int(round(x_range[1]))
    #     y_min, y_max = int(round(y_range[0])), int(round(y_range[1]))

    #     # Możesz też zrobić oddzielne kroki dla X/Y jeśli chcesz
    #     if (x_max - x_min) > 1:
    #         x_min -= delta
    #         x_max += delta
    #     if (y_max - y_min) > 1:
    #         y_min -= delta
    #         y_max += delta

    #     self.setRange(xRange=(x_min, x_max), yRange=(y_min, y_max), padding=0, update=True)
    #     ev.accept()


class SnapImageWidget(pg.GraphicsLayoutWidget):
    def __init__(self):
        super().__init__()
        self.viewbox = PixelSnapViewBox()
        self.viewbox.setAcceptHoverEvents(True)
        self.img_item = pg.ImageItem()
        self.viewbox.addItem(self.img_item)
        self.addItem(self.viewbox)
    def setImage(self, img):
        self.img_item.setImage(img)
    def getView(self):
        return self.viewbox
