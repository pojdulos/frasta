import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator
import sys
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from PyQt5.QtGui import QFont
import matplotlib.pyplot as plt
import time
from functools import wraps

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

    # Siatka pomiarowa
    gx, gy = np.meshgrid(x_vals, y_vals)

    # ----> Dla interpolatora (Y, X)
    query_xy_interp = np.stack([gy.ravel(), gx.ravel()], axis=1)

    # ----> Dla KDTree (X, Y)
    query_xy_tree = np.stack([gx.ravel(), gy.ravel()], axis=1)

    # KDTree zapytanie
    dist, idx = tree.query(query_xy_tree, distance_upper_bound=8.0)

    # Odpowiedzi z scan2
    z2 = np.full(query_xy_tree.shape[0], np.nan)
    valid = idx < len(z)
    z2[valid] = z[idx[valid]]

    # Interpolacja scan1
    interp_func = RegularGridInterpolator((yi, xi), scan1, bounds_error=False, fill_value=np.nan)
    z1 = interp_func(query_xy_interp)

    return z1, z2, x_vals, y_vals

from scipy.ndimage import zoom
import numpy as np

@measure_time
def downsample_scan(scan, xi, yi, factor):
    """
    Zmniejsza skan n-krotnie w X i Y.
    
    Parameters:
        scan: 2D array (Z[y, x])
        xi: 1D array of x coordinates
        yi: 1D array of y coordinates
        factor: downsampling factor (e.g. 2 = 2x mniejsze)
        
    Returns:
        scan_ds: downsampled scan
        xi_ds: downsampled x coordinates
        yi_ds: downsampled y coordinates
    """
    # Zmniejsz siatkę (interpolacja)
    scan_ds = zoom(scan, zoom=(1/factor, 1/factor), order=1)

    # Przeskaluj współrzędne
    xi_ds = np.linspace(xi[0], xi[-1], scan_ds.shape[1])
    yi_ds = np.linspace(yi[0], yi[-1], scan_ds.shape[0])

    return scan_ds, xi_ds, yi_ds

from scipy.ndimage import rotate

class OverlayViewer(pg.GraphicsLayoutWidget):
    def __init__(self, title="Porównanie siatek"):
        super().__init__(title=title)
        self.resize(1400, 500)
        self.setWindowTitle('scan1 | scan2 | różnica')
        self.vmin, self.vmax = -1500, 1500  # wspólna skala kolorów
        self.p1 = self.addPlot(title="scan1")
        self.p3 = self.addPlot(title="różnica (scan1 - scan2)")

        self.rotation_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.rotation_slider.setMinimum(-300)   # odpowiada -30.0°
        self.rotation_slider.setMaximum(300)    # odpowiada +30.0°
        self.rotation_slider.setValue(0)
        self.rotation_slider.setTickInterval(10)
        self.rotation_slider.setSingleStep(1)
        self.rotation_slider.setToolTip("Obrót scan2 (stopnie)")
        self.rotation_slider.valueChanged.connect(self.update_rotation)

        # Układ graficzny (suwak + wykresy)
        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_layout.addWidget(self.rotation_slider)
        self.main_layout.addWidget(self)
        self.container = QtWidgets.QWidget()
        self.container.setLayout(self.main_layout)

        self.container.show()


    def update_rotation(self):
        angle = self.rotation_slider.value() / 10.0  # np. 17 → 1.7°
        
        # Obrót: z zachowaniem kształtu
        rotated_scan2 = rotate(scan2, angle=angle, reshape=False, order=1, mode='nearest')

        # Przelicz różnicę
        z1, z2, x_vals, y_vals = calc_diff_img(scan1, xi, yi, rotated_scan2, xi2, yi2, nx, ny)

        # Różnica tylko tam, gdzie obie wartości są dostępne
        diff = np.full_like(z1, np.nan)
        mask = ~np.isnan(z1) & ~np.isnan(z2)
        diff[mask] = z1[mask] - z2[mask]
        
        diff_img = diff.reshape((ny, nx))
        z1 = z1.reshape((ny, nx))
        z2 = z2.reshape((ny, nx))

        self.update_p1(z1, z2)
        self.update_diff(diff_img)

        # Aktualizuj widoki
        # self.img2.setImage(z2)
        # self.img3.setImage(diff_img)

    def update_p1(self, z1, z2):
        self.p1.clear()  # usuwa stare itemy z wykresu 1

        # --- scan1 ---
        img1 = pg.ImageItem(z1)
        img1.setLevels([self.vmin, self.vmax])
        self.p1.addItem(img1)
        self.p1.setAspectLocked(True)

        # --- scan2 ---
        img2 = pg.ImageItem(z2)
        img2.setLevels([self.vmin, self.vmax])
        img2.setOpacity(.5)
        self.p1.addItem(img2)
        self.p1.setAspectLocked(True)

        self.img2 = img2
        self.img1 = img1


    def update_diff(self, diff_img):
        self.p3.clear()  # usuwa stare itemy z wykresu 3
        # Pobierz colormapę seismic z matplotlib
        cmap_mpl = plt.get_cmap('seismic')
        lut = (cmap_mpl(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
        color_map = pg.ColorMap(pos=np.linspace(0, 1, 256), color=lut)

        # --- różnica ---
        img3 = pg.ImageItem(diff_img)
        img3.setLevels([self.vmin, self.vmax])
        img3.setColorMap(color_map)
        self.p3.addItem(img3)

        self.p3.setAspectLocked(True)

        self.img3 = img3


# Uruchom tylko jeśli ten plik to main
if __name__ == '__main__':
    # # Wczytaj dane z plików .npz
    scan1_data = np.load("source_data/scan1_interp.npz")
    scan1 = scan1_data['grid']

    # Parametry siatki (upewnij się, że są zapisane w plikach)
    pixel_size_x = scan1_data.get('px_x', 0.00138)
    pixel_size_y = scan1_data.get('px_y', 0.00138)
    xi = scan1_data['xi']  # długość = szerokość siatki (X)
    yi = scan1_data['yi']  # długość = wysokość siatki (Y)

    scan1, xi, yi = downsample_scan(scan1, xi, yi, 8)
    pixel_size_x *= 4
    pixel_size_y *= 4
    
    print("scan1 shape:", scan1.shape)
    print("len xi:", len(xi), "| len yi:", len(yi))

    scan2_data = np.load("source_data/scan2_interp.npz")
    scan2 = scan2_data['grid']
    scan2 = np.flipud(scan2)
    scan2 = -scan2

    xi2 = scan2_data['xi']  # oś X dla scan2
    yi2 = scan2_data['yi']  # oś Y dla scan2
    yi2 = yi2[::-1]

    scan2, xi2, yi2 = downsample_scan(scan2, xi2, yi2, 8)

    print("scan2 shape:", scan2.shape)
    print("len xi2:", len(xi2))
    print("len yi2:", len(yi2))


    # Siatka pomiarowa 400x300 w obszarze scan1
    nx, ny = 400, 300

    z1, z2, x_vals, y_vals = calc_diff_img(scan1, xi, yi, scan2, xi2, yi2, nx, ny)

    # Różnica tylko tam, gdzie obie wartości są dostępne
    diff = np.full_like(z1, np.nan)
    mask = ~np.isnan(z1) & ~np.isnan(z2)
    diff[mask] = z1[mask] - z2[mask]
    
    diff_img = diff.reshape((ny, nx))
    z1 = z1.reshape((ny, nx))
    z2 = z2.reshape((ny, nx))

    app = QtWidgets.QApplication(sys.argv)

    viewer = OverlayViewer(title="Porównanie siatek")
    viewer.update_p1(z1, z2)
    viewer.update_diff(diff_img)

    viewer.container.show()
    
    sys.exit(app.exec_())
