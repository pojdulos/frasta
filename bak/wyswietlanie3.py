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


class OverlayViewer(QtWidgets.QWidget):
    def __init__(self, scan1, scan2):
        super().__init__()

        # Layout główny
        layout = QtWidgets.QVBoxLayout(self)

        # Główne okno i scena
        self.view = pg.GraphicsLayoutWidget()
        self.viewbox = self.view.addViewBox()
        self.viewbox.setAspectLocked(True)
        layout.addWidget(self.view)

        # Obrazy
        self.img1 = pg.ImageItem(scan1)
        self.img2 = pg.ImageItem(scan2)
        #self.img2.setOpacity(0.5)

        self.viewbox.addItem(self.img1)
        self.viewbox.addItem(self.img2)

        # Oblicz wspólny zakres jasności z marginesem
        z_min = min(np.nanmin(scan1), np.nanmin(scan2))
        z_max = max(np.nanmax(scan1), np.nanmax(scan2))
        margin = 0.1 * (z_max - z_min)  # 5% margines

        print(z_min, z_max)
        
        levels = (z_min - margin, z_max + margin)
        #levels = (-1000, 1000)

        self.img1.setLevels(levels)
        self.img2.setLevels(levels)


        # Dopasowanie widoku do zawartości
        self.viewbox.autoRange()


        # Suwaki
        self.slider_tx = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_tx.setMinimum(-2000)
        self.slider_tx.setMaximum(2000)
        self.slider_tx.setValue(0)
        self.label_tx = QtWidgets.QLabel("X: 0")

        self.slider_ty = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_ty.setMinimum(-2000)
        self.slider_ty.setMaximum(2000)
        self.slider_ty.setValue(0)
        self.label_ty = QtWidgets.QLabel("Y: 0")

        self.slider_angle = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_angle.setMinimum(-300)
        self.slider_angle.setMaximum(300)
        self.slider_angle.setValue(0)
        self.label_angle = QtWidgets.QLabel("Angle: 0.0°")

        # Layout suwaków z opisami
        layout.addWidget(QtWidgets.QLabel("Translate X"))
        layout.addWidget(self.slider_tx)
        layout.addWidget(self.label_tx)

        layout.addWidget(QtWidgets.QLabel("Translate Y"))
        layout.addWidget(self.slider_ty)
        layout.addWidget(self.label_ty)

        layout.addWidget(QtWidgets.QLabel("Rotate (deg)"))
        layout.addWidget(self.slider_angle)
        layout.addWidget(self.label_angle)

        # Przełączniki
        self.checkbox_visible = QtWidgets.QCheckBox("Pokaż obraz 2")
        self.checkbox_visible.setChecked(True)
        self.checkbox_visible.stateChanged.connect(self.toggleVisibility)

        self.checkbox_blink = QtWidgets.QCheckBox("Migotanie")
        self.checkbox_blink.setChecked(False)
        self.checkbox_blink.stateChanged.connect(self.toggleBlinking)

        layout.addWidget(self.checkbox_visible)
        layout.addWidget(self.checkbox_blink)

        # Timer migotania
        self.blink_timer = QtCore.QTimer()
        self.blink_timer.setInterval(500)
        self.blink_timer.timeout.connect(self.blinkToggle)
        self.blink_state = True

        self.save_button = QtWidgets.QPushButton("Zapisz wyrównane siatki do .h5")
        self.save_button.clicked.connect(self.saveAlignedScans)
        layout.addWidget(self.save_button)

        # zapamiętaj siatki jako atrybuty
        self.original_scan1 = scan1
        self.original_scan2 = scan2


        # Połączenia
        self.slider_tx.valueChanged.connect(self.updateTransform)
        self.slider_ty.valueChanged.connect(self.updateTransform)
        self.slider_angle.valueChanged.connect(self.updateTransform)

        #self.viewbox.setRange(rect=self.img1.boundingRect())

        self.updateTransform()

    def toggleVisibility(self, state):
        # Wyłączenie obrazu drugiego bez migotania
        if not self.checkbox_blink.isChecked():
            self.img2.setVisible(state == QtCore.Qt.Checked)

    def toggleBlinking(self, state):
        if state == QtCore.Qt.Checked:
            self.blink_state = True
            self.blink_timer.start()
        else:
            self.blink_timer.stop()
            self.img2.setVisible(self.checkbox_visible.isChecked())  # przywróć widoczność

    def blinkToggle(self):
        self.blink_state = not self.blink_state
        self.img2.setVisible(self.blink_state)

    def saveAlignedScans(self):
        import h5py
        from PyQt5.QtWidgets import QFileDialog

        # Znajdź wspólny rozmiar
        h = min(self.original_scan1.shape[0], self.original_scan2.shape[0])
        w = min(self.original_scan1.shape[1], self.original_scan2.shape[1])

        aligned1 = self.original_scan1[:h, :w]
        aligned2 = self.original_scan2[:h, :w]

        # Dialog zapisu
        path, _ = QFileDialog.getSaveFileName(self, "Zapisz plik .h5", "aligned.h5", "HDF5 (*.h5)")
        if not path:
            return

        # Zapis do HDF5
        with h5py.File(path, "w") as f:
            f.create_dataset("scan1", data=aligned1)
            f.create_dataset("scan2", data=aligned2)

        QtWidgets.QMessageBox.information(self, "Zapisano", f"Zapisano do pliku:\n{path}")

    def updateTransform(self):
        tx = self.slider_tx.value()
        ty = self.slider_ty.value()
        angle = float(self.slider_angle.value())/10.0

        self.label_tx.setText(f"X: {tx}")
        self.label_ty.setText(f"Y: {ty}")
        self.label_angle.setText(f"Angle: {angle}°")

        transform = QtGui.QTransform()
        transform.translate(tx, ty)
        transform.rotate(angle)

        self.img2.setTransform(transform)



data = np.load("source_data/scan1.npz")
scan1 = data['grid']
x1 = data['x_unique']
y1 = data['y_unique']

data = np.load("source_data/scan2.npz")
scan2 = data['grid']
x2 = data['x_unique']
y2 = data['y_unique']

s1 = np.where(np.isnan(scan1), np.nanmin(scan1), scan1)
s2 = np.where(np.isnan(scan2), np.nanmax(scan2), scan2)

s2 = np.flipud(s2)
s2 = -s2

# Normalizacja
#scan1 = (scan1 - scan1.min()) / (scan1.max() - scan1.min())
#scan2 = (scan2 - scan2.min()) / (scan2.max() - scan2.min())

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    viewer = OverlayViewer(s1, s2)
    viewer.setWindowTitle("Overlay Viewer with Sliders")
    viewer.resize(800, 800)
    viewer.show()
    app.exec_()

