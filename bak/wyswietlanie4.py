import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, shift
import matplotlib.patches as patches
import os
import pandas as pd
import cv2
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
import numpy as np
from scipy.ndimage import affine_transform


import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtGui, QtCore
from scipy.ndimage import affine_transform
import math


import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
from scipy.ndimage import affine_transform
import math


class OverlayViewerRGB(QtWidgets.QWidget):
    def __init__(self, scan1, scan2):
        super().__init__()

        self.raw_scan1 = scan1
        self.raw_scan2 = scan2

        layout = QtWidgets.QVBoxLayout(self)

        # Widok
        self.view = pg.GraphicsLayoutWidget()
        self.viewbox = self.view.addViewBox()
        self.viewbox.setAspectLocked(True)
        layout.addWidget(self.view)

        # Obraz RGB (dynamicznie aktualizowany)
        self.img_rgb = pg.ImageItem()
        self.viewbox.addItem(self.img_rgb)
        self.viewbox.autoRange()

        # Suwaki
        self.slider_tx = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_tx.setRange(-1000, 1000)
        self.slider_tx.setValue(0)
        self.label_tx = QtWidgets.QLabel("X: 0")

        self.slider_ty = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_ty.setRange(-1000, 1000)
        self.slider_ty.setValue(0)
        self.label_ty = QtWidgets.QLabel("Y: 0")

        self.slider_angle = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_angle.setRange(-180, 180)
        self.slider_angle.setValue(0)
        self.label_angle = QtWidgets.QLabel("Angle: 0°")

        layout.addWidget(QtWidgets.QLabel("Translate X"))
        layout.addWidget(self.slider_tx)
        layout.addWidget(self.label_tx)

        layout.addWidget(QtWidgets.QLabel("Translate Y"))
        layout.addWidget(self.slider_ty)
        layout.addWidget(self.label_ty)

        layout.addWidget(QtWidgets.QLabel("Rotate (deg)"))
        layout.addWidget(self.slider_angle)
        layout.addWidget(self.label_angle)

        # Połączenia
        self.slider_tx.valueChanged.connect(self.update_overlay)
        self.slider_ty.valueChanged.connect(self.update_overlay)
        self.slider_angle.valueChanged.connect(self.update_overlay)

        # Pierwsze wywołanie
        self.update_overlay()

    def update_overlay(self):
        tx = self.slider_tx.value()
        ty = self.slider_ty.value()
        angle_deg = self.slider_angle.value()
        angle_rad = math.radians(angle_deg)

        # Aktualizacja etykiet
        self.label_tx.setText(f"X: {tx}")
        self.label_ty.setText(f"Y: {ty}")
        self.label_angle.setText(f"Angle: {angle_deg}°")

        # Macierz obrotu i offset
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        matrix = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
        offset = [-ty, -tx]

        # Transformacja scan2
        scan2_transformed = affine_transform(
            self.raw_scan2,
            matrix,
            offset=offset,
            output_shape=self.raw_scan1.shape,
            order=1,
            mode='constant',
            cval=np.nan
        )

        # Normalizacja i tworzenie RGB
        rgb = self.make_rgb_overlay(self.raw_scan1, scan2_transformed)
        self.img_rgb.setImage(rgb, autoLevels=True)
        self.viewbox.autoRange()


    def make_rgb_overlay(self, scan1, scan2):
        s1 = (scan1 - np.nanmin(scan1)) / (np.nanmax(scan1) - np.nanmin(scan1))
        s2 = (scan2 - np.nanmin(scan2)) / (np.nanmax(scan2) - np.nanmin(scan2))

        s1 = np.nan_to_num(s1, nan=0.0)
        s2 = np.nan_to_num(s2, nan=0.0)

        rgb = np.zeros((*scan1.shape, 3), dtype=np.float32)
        rgb[..., 0] = s1
        rgb[..., 1] = s2
        rgb[..., 2] = (s1 + s2) / 2

        return np.clip(rgb, 0, 1)



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

# Normalizacja
scan1 = (scan1 - scan1.min()) / (scan1.max() - scan1.min())
scan2 = (scan2 - scan2.min()) / (scan2.max() - scan2.min())

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    viewer = OverlayViewerRGB(scan1, scan2)
    viewer.setWindowTitle("Overlay Viewer with Sliders")
    viewer.resize(800, 800)
    viewer.show()
    app.exec_()

