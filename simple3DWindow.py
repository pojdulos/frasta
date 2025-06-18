from pyqtgraph.Qt import QtWidgets, QtGui, QtCore
import numpy as np
import pyqtgraph.opengl as gl


class Simple3DWindow(QtWidgets.QDialog):
    def __init__(self, reference_grid, auto_center_z=True, parent=None):
        super().__init__(parent)
        self.setWindowTitle("3D View of Grids and Profile Plane")
        self.resize(900, 700)
        layout = QtWidgets.QVBoxLayout(self)

        self.view = gl.GLViewWidget()
        layout.addWidget(self.view)
        self.view.setCameraPosition(distance=200)

        # Przechowywane referencje do elementów
        self.surface_ref_item = None

        step = max(1, min(reference_grid.shape[0], reference_grid.shape[1]) // 512)
        # step = 1

        print(f"step={step}")        
        ys = np.arange(0, reference_grid.shape[0], step)
        xs = np.arange(0, reference_grid.shape[1], step)
        Z_ref = reference_grid[np.ix_(ys, xs)]

        Z_MAX = 1e6
        Z_MIN = -1e6

        # Zamień niepoprawne na NaN
        Z_ref = np.where(np.isfinite(Z_ref), Z_ref, np.nan)

        # Zamień punkty spoza zakresu na NaN (nie będą rysowane)
        # Z_ref = np.where((Z_ref > Z_MAX) | (Z_ref < Z_MIN), np.nan, Z_ref)
        # Z_ref = np.where((Z_ref > Z_MAX) | (Z_ref < Z_MIN), 0.0, Z_ref)
        Z_ref = np.where((Z_ref > Z_MAX), Z_MAX, Z_ref)
        Z_ref = np.where((Z_ref < Z_MIN), Z_MIN, Z_ref)

        # Z_ref = Z_ref 

        # Jeśli cała siatka NaN, NIE rysuj jej!
        if not np.all(np.isnan(Z_ref)):
            self.surface_ref_item = gl.GLSurfacePlotItem(
                x=xs, y=ys, z=Z_ref.T, color=(0,1,0,1), shader='shaded'
            )
            self.view.addItem(self.surface_ref_item)

        z_min = np.nanmin(Z_ref)
        z_max = np.nanmax(Z_ref)

        margin = 0.1 * (z_max - z_min)

        z_min -= margin
        z_max += margin

        # Wyznacz bounding box
        all_x = []
        all_y = []
        all_z = []

        if not np.all(np.isnan(Z_ref)):
            all_x.extend(xs)
            all_y.extend(ys)
            all_z.append(np.nanmin(Z_ref))
            all_z.append(np.nanmax(Z_ref))

        # Licz środek sceny
        xc = (min(all_x) + max(all_x)) / 2
        yc = (min(all_y) + max(all_y)) / 2
        zc = (min(all_z) + max(all_z)) / 2 if len(all_z) > 0 else 0

        self.view.setCameraPosition(pos=QtGui.QVector3D(xc, yc, zc))


    def update_data(self, reference_grid):
        """
        Aktualizuje dane w oknie 3D: czyści stare elementy i rysuje nowe na podstawie przekazanych danych.
        """
        self.view.clear()
        self.surface_ref_item = None

        step = max(1, min(reference_grid.shape[0], reference_grid.shape[1]) // 512)
        ys = np.arange(0, reference_grid.shape[0], step)
        xs = np.arange(0, reference_grid.shape[1], step)
        Z_ref = reference_grid[np.ix_(ys, xs)]

        Z_MAX = 1e6
        Z_MIN = -1e6
        Z_ref = np.where(np.isfinite(Z_ref), Z_ref, np.nan)
        Z_ref = np.where((Z_ref > Z_MAX) | (Z_ref < Z_MIN), 0.0, Z_ref)

        if not np.all(np.isnan(Z_ref)):
            self.surface_ref_item = gl.GLSurfacePlotItem(
                x=xs, y=ys, z=Z_ref.T, color=(0,1,0,1), shader='shaded'
            )
            self.view.addItem(self.surface_ref_item)

        z_min = np.nanmin(Z_ref)
        z_max = np.nanmax(Z_ref)
        z_min -= np.abs(0.1 * z_min)
        z_max += np.abs(0.1 * z_max)

        # Ustaw kamerę na środek nowych danych
        all_x = []
        all_y = []
        all_z = []
        if not np.all(np.isnan(Z_ref)):
            all_x.extend(xs)
            all_y.extend(ys)
            all_z.append(np.nanmin(Z_ref))
            all_z.append(np.nanmax(Z_ref))

        xc = (min(all_x) + max(all_x)) / 2
        yc = (min(all_y) + max(all_y)) / 2
        zc = (min(all_z) + max(all_z)) / 2 if len(all_z) > 0 else 0

        self.view.setCameraPosition(pos=QtGui.QVector3D(xc, yc, zc))

