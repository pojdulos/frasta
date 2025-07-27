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
        # self.view.setCameraPosition(distance=200)

        # Przechowywane referencje do elementów
        self.surface_ref_item = None
        self.scatter = None

        step = max(1, min(reference_grid.shape[0], reference_grid.shape[1]) // 512)
        # step = 1

        print(f"step={step}")        
        ys = np.arange(0, reference_grid.shape[0], step)
        xs = np.arange(0, reference_grid.shape[1], step)
        Z_ref = reference_grid[np.ix_(ys, xs)]

        Z_MAX = 1e6
        Z_MIN = -1e6

        Z_ref = np.where(np.isfinite(Z_ref), Z_ref, np.nan)

        Z_ref = np.where((Z_ref > Z_MAX), Z_MAX, Z_ref)
        Z_ref = np.where((Z_ref < Z_MIN), Z_MIN, Z_ref)
 
        if not np.all(np.isnan(Z_ref)):
            mask = np.isfinite(Z_ref)
            yy, xx = np.meshgrid(ys, xs, indexing='ij')
            #points = np.column_stack((xx[mask], yy[mask], Z_ref[mask])).astype(np.float32)

            N = 1000
            points = np.random.normal(size=(N,3)) * 10 + 50
            points = points.astype(np.float32)

            center = np.mean(points, axis=0)
            scene_size = np.linalg.norm(np.max(points, axis=0) - np.min(points, axis=0))
            print("center:", center)
            print("scene_size:", scene_size)

            # Przesuwamy całą chmurę na środek sceny
            points_centered = points - center

            self.scatter = gl.GLScatterPlotItem(
                pos=points_centered[:500],
                color=(1, 1, 0, 1),
                size=50,
                pxMode=True
            )
            self.view.addItem(self.scatter)
            self.view.setCameraPosition(pos=QtGui.QVector3D(0,0,0), distance=scene_size * 1.2)

