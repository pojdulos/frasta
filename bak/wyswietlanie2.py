import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, shift
import matplotlib.patches as patches
import os
import pandas as pd
import cv2
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
import numpy as np

data = np.load("source_data/scan1.npz")
scan1 = data['grid']
x1 = data['x_unique']
y1 = data['y_unique']

data = np.load("source_data/scan2.npz")
scan2 = data['grid']
x2 = data['x_unique']
y2 = data['y_unique']

scan1 = np.where(np.isnan(scan1), np.nanmin(scan1), scan1)
scan2 = np.where(np.isnan(scan2), np.nanmax(scan2), scan2)

scan2 = np.flipud(scan2)
scan2 = -scan2


# Start aplikacji
app = QtWidgets.QApplication([])

# Główne okno z poziomym layoutem
main_window = QtWidgets.QWidget()
layout = QtWidgets.QHBoxLayout(main_window)

# Tworzymy dwa widoki
view1 = pg.ImageView()
view2 = pg.ImageView()

# Ustawiamy obrazy i zakresy jasności
view1.setImage(scan1, levels=(scan1.min(), scan1.max()))
view2.setImage(scan2, levels=(scan2.min(), scan2.max()))

# Dodajemy widoki do layoutu
layout.addWidget(view1)
layout.addWidget(view2)

# Pokaż okno
main_window.show()
app.exec_()