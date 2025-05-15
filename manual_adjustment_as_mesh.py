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



def grid_to_mesh_slow(grid, pixel_size_x=1.0, pixel_size_y=1.0):
    """
    Konwertuje siatkę 2D (Z) na mesh trójkątów w 3D.
    """
    h, w = grid.shape
    vertices = []
    faces = []
    vertex_map = {}

    def get_index(ix, iy):
        key = (ix, iy)
        if key not in vertex_map:
            z = grid[iy, ix]
            x = ix * pixel_size_x
            y = iy * pixel_size_y
            vertex_map[key] = len(vertices)
            vertices.append((x, y, z))
        return vertex_map[key]

    for iy in range(h - 1):
        for ix in range(w - 1):
            # Kwadrat: (ix, iy), (ix+1, iy), (ix, iy+1), (ix+1, iy+1)
            z00 = grid[iy, ix]
            z10 = grid[iy, ix+1]
            z01 = grid[iy+1, ix]
            z11 = grid[iy+1, ix+1]

            # Jeśli jakikolwiek punkt NaN, pomiń
            if np.any(np.isnan([z00, z10, z01, z11])):
                continue

            # Wierzchołki
            i00 = get_index(ix, iy)
            i10 = get_index(ix+1, iy)
            i01 = get_index(ix, iy+1)
            i11 = get_index(ix+1, iy+1)

            # Dwa trójkąty na kwadrat
            faces.append((i00, i10, i11))
            faces.append((i00, i11, i01))

    return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)

def grid_to_mesh(grid, pixel_size_x=1.0, pixel_size_y=1.0):
    h, w = grid.shape

    # Siatka XY
    y_indices, x_indices = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    x_coords = x_indices * pixel_size_x
    y_coords = y_indices * pixel_size_y
    z_coords = grid

    # Wszystkie wierzchołki
    vertices = np.stack([x_coords, y_coords, z_coords], axis=-1).reshape(-1, 3)

    # Maska ważnych punktów (nie NaN)
    valid_mask = ~np.isnan(vertices[:, 2])
    index_map = -np.ones(h * w, dtype=int)
    index_map[valid_mask] = np.arange(np.count_nonzero(valid_mask))

    # Indeksy trójkątów
    idx_tl = np.ravel_multi_index((np.arange(h - 1)[:, None], np.arange(w - 1)[None, :]), dims=(h, w))
    idx_tr = idx_tl + 1
    idx_bl = idx_tl + w
    idx_br = idx_bl + 1

    # Spłaszczone i połączone
    idx_tl = idx_tl.ravel()
    idx_tr = idx_tr.ravel()
    idx_bl = idx_bl.ravel()
    idx_br = idx_br.ravel()

    # Tylko tam, gdzie wszystkie 4 są ważne
    valid_quad = (index_map[idx_tl] >= 0) & (index_map[idx_tr] >= 0) & \
                 (index_map[idx_bl] >= 0) & (index_map[idx_br] >= 0)

    # Dwa trójkąty na każdy kwadrat
    faces_a = np.stack([index_map[idx_tl], index_map[idx_tr], index_map[idx_br]], axis=1)[valid_quad]
    faces_b = np.stack([index_map[idx_tl], index_map[idx_br], index_map[idx_bl]], axis=1)[valid_quad]
    faces = np.vstack([faces_a, faces_b])

    # Przefiltrowane wierzchołki
    vertices = vertices[valid_mask]

    return vertices.astype(np.float32), faces.astype(np.int32)

import trimesh
import numpy as np
import trimesh

def raycast_diff_map_sparse(mesh1, mesh2, 
                            x_range, y_range,
                            grid_resolution_x=400,
                            grid_resolution_y=300,
                            ray_direction=np.array([0, 0, -1]),
                            ray_origin_z=1e5):
    """
    Zwraca rzadki obraz różnicowy (Z1 - Z2) oparty na rzutowaniu promieni w dół.

    mesh1, mesh2: trimesh.Trimesh
    x_range, y_range: tuple (min, max) w jednostkach siatki
    grid_resolution_x/y: liczba pikseli w X i Y
    """
    xs = np.linspace(x_range[0], x_range[1], grid_resolution_x)
    ys = np.linspace(y_range[0], y_range[1], grid_resolution_y)
    gx, gy = np.meshgrid(xs, ys)

    n_points = gx.size
    origins = np.stack([gx.ravel(), gy.ravel(), np.full(n_points, ray_origin_z)], axis=1)
    directions = np.tile(ray_direction, (n_points, 1))

    z1 = np.full(n_points, np.nan)
    z2 = np.full(n_points, np.nan)

    loc1, idx1, _ = mesh1.ray.intersects_location(origins, directions)
    loc2, idx2, _ = mesh2.ray.intersects_location(origins, directions)

    for i, pt in zip(idx1, loc1):
        z1[i] = pt[2]
    for i, pt in zip(idx2, loc2):
        z2[i] = pt[2]

    diff = z1 - z2
    return gx, gy, diff.reshape((grid_resolution_y, grid_resolution_x))

data = np.load("source_data/mesh1_data.npz")
v1 = data["vertices"]
f1 = data["faces"]

data = np.load("source_data/mesh2_data.npz")
v2 = data["vertices"]
f2 = data["faces"]

mesh1 = trimesh.Trimesh(vertices=v1, faces=f1, process=False)
# mesh1.export("mesh1_output.obj")
mesh1.ray = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh1)

mesh2 = trimesh.Trimesh(vertices=v2, faces=f2, process=False)
# mesh2.export("mesh2_output.obj")
mesh2.ray = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh2)


gx, gy, diff_img = raycast_diff_map_sparse(
    mesh1, mesh2,
    x_range=(300, 8000),
    y_range=(300, 6000),
    grid_resolution_x=400,
    grid_resolution_y=300
)

import matplotlib.pyplot as plt

plt.imshow(diff_img, cmap='seismic', origin='lower',
           extent=[gx.min(), gx.max(), gy.min(), gy.max()])
plt.colorbar(label="Z difference [units]")
plt.title("Sparse difference map (raycast)")
plt.show()


# #sigma = 0.5
# #scan1 = gaussian_filter(scan1, sigma=sigma)

# s1 = scan1 #np.where(np.isnan(scan1), np.nanmin(scan1), scan1)
# s2 = scan2 # np.where(np.isnan(scan2), np.nanmax(scan2), scan2)

# s2 = np.flipud(s2)
# s2 = -s2

# # Normalizacja
# #scan1 = (scan1 - scan1.min()) / (scan1.max() - scan1.min())
# #scan2 = (scan2 - scan2.min()) / (scan2.max() - scan2.min())

# if __name__ == '__main__':
#     app = QtWidgets.QApplication([])
#     viewer = OverlayViewer(s1, s2)
#     viewer.setWindowTitle("Overlay Viewer with Sliders")
#     viewer.resize(800, 800)
#     viewer.show()
#     app.exec_()

