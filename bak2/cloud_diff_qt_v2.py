import numpy as np
import sys
import time
from functools import wraps
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import zoom, rotate
import matplotlib.pyplot as plt
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore


def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f">>> {func.__name__}() took {end - start:.4f} seconds")
        return result
    return wrapper


@measure_time
def calc_diff_img(scan1, xi, yi, scan2, xi2, yi2, nx, ny):
    x2, y2 = np.meshgrid(xi2, yi2)
    points_xyz = np.stack([x2.ravel(), y2.ravel(), scan2.ravel()], axis=1)
    points_xyz = points_xyz[np.isfinite(points_xyz).all(axis=1)]

    xy = points_xyz[:, :2]
    z = points_xyz[:, 2]
    tree = cKDTree(xy)

    x_vals = np.linspace(xi.min(), xi.max(), nx)
    y_vals = np.linspace(yi.min(), yi.max(), ny)
    gx, gy = np.meshgrid(x_vals, y_vals)

    query_xy_interp = np.stack([gy.ravel(), gx.ravel()], axis=1)
    query_xy_tree = np.stack([gx.ravel(), gy.ravel()], axis=1)

    dist, idx = tree.query(query_xy_tree, distance_upper_bound=8.0)

    z2 = np.full(query_xy_tree.shape[0], np.nan)
    valid = idx < len(z)
    z2[valid] = z[idx[valid]]

    interp_func = RegularGridInterpolator((yi, xi), scan1, bounds_error=False, fill_value=np.nan)
    z1 = interp_func(query_xy_interp)

    return z1.reshape((ny, nx)), z2.reshape((ny, nx)), x_vals, y_vals


@measure_time
def downsample_scan(scan, xi, yi, factor):
    scan_ds = zoom(scan, zoom=(1/factor, 1/factor), order=1)
    xi_ds = np.linspace(xi[0], xi[-1], scan_ds.shape[1])
    yi_ds = np.linspace(yi[0], yi[-1], scan_ds.shape[0])
    return scan_ds, xi_ds, yi_ds


class OverlayViewer(QtWidgets.QWidget):
    def __init__(self, scan1, xi, yi, scan2, xi2, yi2, nx, ny):
        super().__init__()
        self.setWindowTitle("Porównanie siatek")
        self.vmin, self.vmax = -1500, 1500

        self.scan1, self.xi, self.yi = scan1, xi, yi
        self.base_scan2, self.xi2, self.yi2 = scan2, xi2, yi2
        self.nx, self.ny = nx, ny

        self.layout = QtWidgets.QVBoxLayout(self)

        self.rotation_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.rotation_slider.setMinimum(-300)
        self.rotation_slider.setMaximum(300)
        self.rotation_slider.setValue(0)
        self.rotation_slider.setTickInterval(10)
        self.rotation_slider.setSingleStep(1)
        self.rotation_slider.setToolTip("Obrót scan2 (stopnie)")
        self.rotation_slider.valueChanged.connect(self.update_rotation)

        self.layout.addWidget(self.rotation_slider)

        self.graph_widget = pg.GraphicsLayoutWidget()
        self.p1 = self.graph_widget.addPlot(title="scan1 + scan2")
        self.p1.setAspectLocked(True)
        #self.graph_widget.nextRow()
        self.p3 = self.graph_widget.addPlot(title="różnica (scan1 - scan2)")
        self.p3.setAspectLocked(True)
        self.layout.addWidget(self.graph_widget)

        self.update_rotation()

    def update_rotation(self):
        angle = self.rotation_slider.value() / 10.0
        rotated_scan2 = rotate(self.base_scan2, angle=angle, reshape=False, order=1, mode='nearest')
        z1, z2, *_ = calc_diff_img(self.scan1, self.xi, self.yi, rotated_scan2, self.xi2, self.yi2, self.nx, self.ny)

        diff = np.full_like(z1, np.nan)
        mask = ~np.isnan(z1) & ~np.isnan(z2)
        diff[mask] = z1[mask] - z2[mask]

        self.update_p1(z1, z2)
        self.update_diff(diff)

    def update_p1(self, z1, z2):
        self.p1.clear()
        img1 = pg.ImageItem(z1)
        img1.setLevels([self.vmin, self.vmax])
        self.p1.addItem(img1)

        img2 = pg.ImageItem(z2)
        img2.setLevels([self.vmin, self.vmax])
        img2.setOpacity(0.5)
        self.p1.addItem(img2)

    def update_diff(self, diff_img):
        self.p3.clear()
        cmap = plt.get_cmap('seismic')
        lut = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
        color_map = pg.ColorMap(np.linspace(0, 1, 256), lut)

        img3 = pg.ImageItem(diff_img)
        img3.setLevels([self.vmin, self.vmax])
        img3.setColorMap(color_map)
        self.p3.addItem(img3)


if __name__ == '__main__':
    scan1_data = np.load("source_data/scan1_interp.npz")
    scan1 = scan1_data['grid']
    xi = scan1_data['xi']
    yi = scan1_data['yi']

    scan2_data = np.load("source_data/scan2_interp.npz")
    scan2 = np.flipud(-scan2_data['grid'])
    xi2 = scan2_data['xi']
    yi2 = scan2_data['yi'][::-1]

    scan1, xi, yi = downsample_scan(scan1, xi, yi, 8)
    scan2, xi2, yi2 = downsample_scan(scan2, xi2, yi2, 8)

    nx, ny = 400, 300

    app = QtWidgets.QApplication(sys.argv)
    viewer = OverlayViewer(scan1, xi, yi, scan2, xi2, yi2, nx, ny)
    viewer.resize(1400, 800)
    viewer.show()
    sys.exit(app.exec_())
