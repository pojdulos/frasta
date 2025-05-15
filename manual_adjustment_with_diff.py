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
from scipy.ndimage import gaussian_filter

class OverlayViewer(QtWidgets.QWidget):
    def __init__(self, scan1, scan2):
        super().__init__()

        # Layout główny
        layout = QtWidgets.QVBoxLayout(self)

        # Główne okno i scena
        self.view = pg.GraphicsLayoutWidget()
        self.viewbox = self.view.addViewBox()
        self.viewbox.setAspectLocked(True)
        
        self.scan1 = scan1
        self.scan2 = scan2

        # Obrazy
        self.img1 = pg.ImageItem(scan1)
        self.img2 = pg.ImageItem(scan2)
        self.img2.setOpacity(0.5)

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


        # Obraz różnicy
        self.diff_view = pg.ImageView(view=pg.PlotItem())
        self.diff_view.ui.histogram.hide()
        self.diff_view.ui.roiBtn.hide()
        self.diff_view.ui.menuBtn.hide()
        self.diff_view.setMinimumWidth(300)

        # Układ poziomy: widok główny + widok różnic
        split = QtWidgets.QHBoxLayout()
        split.addWidget(self.view)
        split.addWidget(self.diff_view)
        layout.addLayout(split)

        south_layout = QtWidgets.QHBoxLayout()

        south_layout.addLayout(self.sliders_layout())

        # Przełączniki
        self.checkbox_visible = QtWidgets.QCheckBox("scan2 is visible")
        self.checkbox_visible.setChecked(True)
        self.checkbox_visible.stateChanged.connect(self.toggleVisibility)

        self.checkbox_trans = QtWidgets.QCheckBox("scan2 translucence")
        self.checkbox_trans.setChecked(True)
        self.checkbox_trans.stateChanged.connect(self.toggleTransparency)

        self.checkbox_blink = QtWidgets.QCheckBox("scan2 blinking")
        self.checkbox_blink.setChecked(False)
        self.checkbox_blink.stateChanged.connect(self.toggleBlinking)

        tool2_layout = QtWidgets.QVBoxLayout()

        tool2_layout.addWidget(self.checkbox_visible)
        tool2_layout.addWidget(self.checkbox_trans)
        tool2_layout.addWidget(self.checkbox_blink)

        # Timer migotania
        self.blink_timer = QtCore.QTimer()
        self.blink_timer.setInterval(500)
        self.blink_timer.timeout.connect(self.blinkToggle)
        self.blink_state = True

        self.update_diff_btn = QtWidgets.QPushButton("Aktualizuj obraz różnic")
        self.update_diff_btn.clicked.connect(self.update_difference_map)
        tool2_layout.addWidget(self.update_diff_btn)

        self.save_button = QtWidgets.QPushButton("Save aligned grids to .h5")
        self.save_button.clicked.connect(self.saveAlignedScans)
        tool2_layout.addWidget(self.save_button)

        south_layout.addLayout(tool2_layout)

        layout.addLayout(south_layout)

        # zapamiętaj siatki jako atrybuty
        self.original_scan1 = scan1
        self.original_scan2 = scan2


        # Połączenia
        self.slider_tx.valueChanged.connect(self.updateTransform)
        self.slider_ty.valueChanged.connect(self.updateTransform)
        self.slider_angle.valueChanged.connect(self.updateTransform)

        #self.viewbox.setRange(rect=self.img1.boundingRect())

        self.updateTransform()

    def sliders_layout(self):
        # Suwaki
        self.slider_tx = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        #self.slider_tx = QtWidgets.QSpinBox()
        self.slider_tx.setMinimum(-2000)
        self.slider_tx.setMaximum(2000)
        self.slider_tx.setValue(0)
        self.label_tx = QtWidgets.QLabel("X: 0")
        self.label_tx.setMinimumWidth(70)

        self.slider_ty = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        #self.slider_ty = QtWidgets.QSpinBox()
        self.slider_ty.setMinimum(-2000)
        self.slider_ty.setMaximum(2000)
        self.slider_ty.setValue(0)
        self.label_ty = QtWidgets.QLabel("Y: 0")
        self.label_ty.setMinimumWidth(70)

        self.slider_angle = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_angle.setMinimum(-300)
        self.slider_angle.setMaximum(300)
        self.slider_angle.setValue(0)
        self.label_angle = QtWidgets.QLabel("Angle: 0.0°")
        self.label_angle.setMinimumWidth(70)

        tool_layout = QtWidgets.QVBoxLayout()
        
        # Layout suwaków z opisami
        trx_layout = QtWidgets.QHBoxLayout()
        # trx_layout.addWidget(QtWidgets.QLabel("Translate X"))
        trx_layout.addWidget(self.label_tx)
        trx_layout.addWidget(self.slider_tx)
        tool_layout.addLayout(trx_layout)

        try_layout = QtWidgets.QHBoxLayout()
        # try_layout.addWidget(QtWidgets.QLabel("Translate Y"))
        try_layout.addWidget(self.label_ty)
        try_layout.addWidget(self.slider_ty)
        tool_layout.addLayout(try_layout)

        rot_layout = QtWidgets.QHBoxLayout()
        # rot_layout.addWidget(QtWidgets.QLabel("Rotate (deg)"))
        rot_layout.addWidget(self.label_angle)
        rot_layout.addWidget(self.slider_angle)
        tool_layout.addLayout(rot_layout)
        
        return tool_layout


    def toggleVisibility(self, state):
        # Wyłączenie obrazu drugiego bez migotania
        if not self.checkbox_blink.isChecked():
            self.img2.setVisible(state == QtCore.Qt.Checked)

    def toggleTransparency(self, state):
        if state == QtCore.Qt.Checked:
            self.img2.setOpacity(0.5)
        else:
            self.img2.setOpacity(1.0)

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

    

    def update_difference_map(self):
        tx = self.slider_tx.value()
        ty = self.slider_ty.value()

        from scipy.ndimage import affine_transform

        # QTransform do macierzy 3x3
        qt_transform = self.img2.transform()
        m = np.array([
            [qt_transform.m11(), qt_transform.m21(), qt_transform.m31()],
            [qt_transform.m12(), qt_transform.m22(), qt_transform.m32()],
            [0,                 0,                 1]
        ])

        # Zamień na 2x3 do affine_transform (odwrócona!)
        affine_matrix = np.linalg.inv(m)[0:2, 0:3]

        # Przekształć obraz
        scan2_trans = affine_transform(
            self.scan2,
            matrix=affine_matrix[:, :2],
            offset=affine_matrix[:, 2],
            order=1,
            mode='constant',
            cval=np.nan
        )

        scan_diff = self.scan1 - scan2_trans
        self.diff_view.setImage(np.nan_to_num(scan_diff, nan=0), autoLevels=False)

        # angle = float(self.slider_angle.value()) / 10.0

        # # Przekształcenie danych
        # img2_rotated = rotate(self.original_scan2, angle, reshape=False, order=1, mode='constant', cval=np.nan)
        # img2_transformed = shift(img2_rotated, shift=(ty, tx), order=1, mode='constant', cval=np.nan)

        # self.diff_view.setImage(np.nan_to_num(distance_map, nan=0), autoLevels=False)

        # h = min(img2_transformed.shape[0], img1_data.shape[0])
        # w = min(img2_transformed.shape[1], img1_data.shape[1])
        # img2_transformed = img2_transformed[:h, :w]
        # img1_data = img1_data[:h, :w]

        # distance_map = img1_data - img2_transformed
        # valid_mask = ~np.isnan(img1_data) & ~np.isnan(img2_transformed)
        # distance_map = np.where(valid_mask, distance_map, np.nan)

        image_item = self.diff_view.getImageItem()
        lut = pg.colormap.get("inferno").getLookupTable(0.0, 1.0, 512)
        image_item.setLookupTable(lut)

        # self.diff_view.setImage(np.nan_to_num(distance_map, nan=0), autoLevels=False)
        # self.diff_view.setLevels(-500, 500)

    def updateTransform(self):
        tx = self.slider_tx.value()
        ty = self.slider_ty.value()
        angle = float(self.slider_angle.value())/10.0

        self.label_tx.setText(f"X: {tx}")
        self.label_ty.setText(f"Y: {ty}")
        self.label_angle.setText(f"Angle: {angle}°")

        h, w = self.original_scan2.shape
        cx, cy = w / 2, h / 2

        transform = QtGui.QTransform()
        transform.translate(tx + cx, ty + cy)  # przesuń do środka
        transform.rotate(angle)                # obrót wokół środka
        transform.translate(-cx, -cy)          # wróć na miejsce

        # transform = QtGui.QTransform()
        # transform.translate(tx, ty)
        # transform.rotate(angle)

        self.img2.setTransform(transform)

        # # Oblicz przekształcony obraz
        # img2_transformed = self.img2.image
        # img1_data = self.img1.image

        # if img2_transformed is None or img1_data is None:
        #     return

        # # 1. Oblicz transformację jako macierz
        # angle_rad = np.deg2rad(angle)
        # cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        # # 2. Oblicz transformację odwrotną i zastosuj do siatki
        # # użyj scipy.ndimage.shift + rotate
        # img2_transformed = rotate(self.original_scan2, angle, reshape=False, order=1, mode='constant', cval=np.nan)
        # img2_transformed = shift(img2_transformed, shift=(ty, tx), order=1, mode='constant', cval=np.nan)

        # h = min(img2_transformed.shape[0], img1_data.shape[0])
        # w = min(img2_transformed.shape[1], img1_data.shape[1])
        # img2_transformed = img2_transformed[:h, :w]
        # img1_data = img1_data[:h, :w]

        # distance_map = img1_data - img2_transformed
        # valid_mask = ~np.isnan(img1_data) & ~np.isnan(img2_transformed)
        # distance_map = np.where(valid_mask, distance_map, np.nan)

        # # Uzyskaj dostęp do ImageItem w ImageView
        # image_item = self.diff_view.getImageItem()

        # # Ustaw mapę kolorów
        # lut = pg.colormap.get("plasma").getLookupTable(0.0, 1.0, 512)
        # image_item.setLookupTable(lut)

        # # Teraz możesz bezpiecznie ustawić obraz
        # self.diff_view.setImage(np.nan_to_num(distance_map, nan=0), autoLevels=False)
        # self.diff_view.setLevels(-500, 500)



data = np.load("source_data/scan1_interp.npz")
scan1 = data['grid']
# x1 = data['x_unique']
# y1 = data['y_unique']

data = np.load("source_data/scan2_interp.npz")
scan2 = data['grid']
# x2 = data['x_unique']
# y2 = data['y_unique']

#sigma = 0.5
#scan1 = gaussian_filter(scan1, sigma=sigma)

s1 = scan1 #np.where(np.isnan(scan1), np.nanmin(scan1), scan1)
s2 = scan2 # np.where(np.isnan(scan2), np.nanmax(scan2), scan2)

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

