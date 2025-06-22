from pyqtgraph.Qt import QtWidgets, QtGui, QtCore
import numpy as np
import pyqtgraph.opengl as gl


import numpy as np
from scipy.interpolate import griddata

def fill_holes(grid):
    if grid is None:
        return

    tst = np.isnan(grid)
    
    grid_x, grid_y = np.meshgrid(np.arange(grid.shape[1]), np.arange(grid.shape[0]))  # tworzymy siatkę 1x1

    interp_points = np.column_stack((grid_x[tst], grid_y[tst]))

    valid = ~np.isnan(grid)
    interp_values = griddata(
        (grid_x[valid], grid_y[valid]),
        grid[valid],
        interp_points,
        method='nearest'  # lub 'linear', jeśli wolisz łagodniejsze przejścia
    )

    grid[tst] = interp_values
    return grid

def remove_outliers(original_grid, smoothed_grid, threshold):
    """
    Zamienia outliery w original_grid na wartości z siatki wygładzonej,
    jeśli różnica przekracza próg (threshold).
    """
    diff = np.abs(original_grid - smoothed_grid)
    mask_outlier = diff > threshold

    cleaned = original_grid.copy()
    # cleaned[mask_outlier] = np.nan
    cleaned[mask_outlier] = smoothed_grid[mask_outlier]
    return cleaned


class Simple3DWindow(QtWidgets.QDialog):
    def __init__(self, grid, parent=None):
        super().__init__(parent)
        self.setWindowTitle("3D View of Grids and Profile Plane")
        self.resize(900, 700)
        layout = QtWidgets.QVBoxLayout(self)

        self.view = gl.GLViewWidget()
        layout.addWidget(self.view)
        self.view.setCameraPosition(distance=200)

        # Przechowywane referencje do elementów
        self.surface_ref_item = None

        self.update_data(grid)


    def update_data(self, grid_original):
        grid_filled = fill_holes(grid_original)
        
        from scipy.ndimage import gaussian_filter
        sigma = 25
        grid_smooth = gaussian_filter(grid_filled, sigma)
       
        threshold = 100
        grid = remove_outliers(grid_original, grid_smooth, threshold)

        self.view.clear()
        #self.surface_ref_item = None

        step = max(1, min(grid.shape[0], grid.shape[1]) // 512)
        ys = np.arange(0, grid.shape[0], step)
        xs = np.arange(0, grid.shape[1], step)
        Z_ref = grid[np.ix_(ys, xs)]

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

