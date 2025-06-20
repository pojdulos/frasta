import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from skimage.segmentation import flood
from scipy.interpolate import griddata
import trimesh
import time
from functools import wraps

from responsiveInfiniteLine import ResponsiveInfiniteLine
from gridData import GridData

def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f">>> {func.__name__}() took {end - start:.4f} seconds")
        return result
    return wrapper

class ScanTab(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_view = pg.ImageView()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        self.image_view.ui.histogram.hide()
        self.image_view.getView().setMenuEnabled(False)

        self.hist_widget = pg.PlotWidget()
        self.hist_widget.setMaximumHeight(120)  # Wąski pasek
        self.hist_widget.setMenuEnabled(False)
        self.hist_widget.setMouseEnabled(x=False, y=False)

        hlayout = QtWidgets.QVBoxLayout(self)
        hlayout.addWidget(self.image_view, stretch=1)
        hlayout.addWidget(self.hist_widget)

        self.setLayout(hlayout)

        self.zero_point_mode = False
        self.tilt_mode = False

        self.seed_points = []
        self.grid = None
        self.masked = None
        self.xi = None
        self.yi = None
        self.px_x = None
        self.px_y = None

        self.is_colormap = False
        self.current_colormap = 'gray'  # lub None

        self.zero_window_size = 15  # lub inna liczba nieparzysta
        self.zero_sigma = 2.0       # ile odchyleń przyjmujesz jako "nie odstające"

        self.image_view.getView().scene().sigMouseClicked.connect(self.mouse_clicked)


    def update_histogram(self):
        if self.grid is None:
            return
        data = self.grid[~np.isnan(self.grid)]
        if data.size == 0:
            self.hist_widget.clear()
            return

        y, x = np.histogram(data, bins=1024)
        self.hist_widget.clear()
        self.hist_plot = self.hist_widget.plot(x, y, stepMode=True, fillLevel=0, brush=(150, 150, 150, 150))

        # Zapamiętaj stare pozycje (jeśli istnieją)
        vmin = float(np.min(data))
        vmax = float(np.max(data))
        min_line_pos = getattr(self, 'hist_min_line', None)
        max_line_pos = getattr(self, 'hist_max_line', None)
        min_val = min_line_pos.value() if min_line_pos else vmin
        max_val = max_line_pos.value() if max_line_pos else vmax

        # Pozycje linii nie mogą wyjść poza nowe dane!
        min_val = max(min_val, vmin)
        max_val = min(max_val, vmax)

        self.hist_min_line = ResponsiveInfiniteLine(update_callback=self.update_histogram_threshold, angle=90, movable=True, pen=pg.mkPen('b', width=2), hoverPen=pg.mkPen('y', width=2))
        self.hist_max_line = ResponsiveInfiniteLine(update_callback=self.update_histogram_threshold, angle=90, movable=True, pen=pg.mkPen('r', width=2), hoverPen=pg.mkPen('y', width=2))
        self.hist_widget.addItem(self.hist_min_line)
        self.hist_widget.addItem(self.hist_max_line)
        self.hist_min_line.setValue(min_val)
        self.hist_max_line.setValue(max_val)

    def on_hist_line_changed(self):
        vmin = min(self.hist_min_line.value(), self.hist_max_line.value())
        vmax = max(self.hist_min_line.value(), self.hist_max_line.value())
        self.update_image(vmin, vmax)

    def update_histogram_threshold(self, value):
        # print("Zaktualizowano próg:", value)
        vmin = min(self.hist_min_line.value(), self.hist_max_line.value())
        vmax = max(self.hist_min_line.value(), self.hist_max_line.value())
        self.update_image(vmin, vmax)

    def set_zero_point_mode(self):
        self.zero_point_mode = True
        # QtWidgets.QMessageBox.information(self, "Wybierz punkt", "Kliknij na widoku skanu punkt, który ma być nowym zerem.")

    def set_tilt_mode(self):
        self.tilt_mode = True
        # QtWidgets.QMessageBox.information(self, "Wybierz punkt", "Kliknij na widoku skanu punkt, który ma być nowym zerem.")

    def set_mask(self, mask, inside=True):
        if self.grid is not None:
            self.grid = np.where(mask, self.grid, np.nan)
            self.update_image()
            self.update_histogram()

    def getGridData(self):
        data = GridData(
            self.grid,
            self.xi,
            self.yi,
            self.px_x,
            self.px_y,
            float(np.min(self.grid)),
            float(np.max(self.grid))
        )
        if hasattr(self, 'hist_min_line') and hasattr(self, 'hist_max_line'):
            data.vmin = min(self.hist_min_line.value(), self.hist_max_line.value())
            data.vmax = max(self.hist_min_line.value(), self.hist_max_line.value())
        return data
    
    def setGridData(self, data: GridData):
        self.grid = data.grid
        self.xi = data.xi
        self.yi = data.yi
        self.px_x = data.px_x
        self.px_y = data.px_y
        self.update_image()
        self.update_histogram()
        self.hist_min_line.setValue(data.vmin)
        self.hist_max_line.setValue(data.vmax)

    
    # def get_data(self):
    #     return {
    #         'grid': self.grid,
    #         'xi': self.xi,
    #         'yi': self.yi,
    #         'px_x': self.px_x,
    #         'px_y': self.px_y
    #     }


    def set_data(self, grid, xi, yi, px_x, px_y):
        self.grid = grid
        self.masked = grid
        self.xi = xi
        self.yi = yi
        self.px_x = px_x
        self.px_y = px_y
        #print(f"grid: {self.grid.shape}, xmin: {self.xi[0]}, ymin: {self.yi[0]}, px_x: {self.px_x}, px_y: {self.px_y}")
        self.update_image()
        self.update_histogram()


    def grid_to_mesh_vectorized(self, grid, pixel_size_x=1.0, pixel_size_y=1.0):
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


    def save_as_mesh(self, grid, px_x=1.38, px_y=1.38):
        v, f = self.grid_to_mesh_vectorized(grid, px_x, px_y)
        mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
        mesh.export("mesh_output.obj")


    def flip_scan(self, parent=None):
        if self.grid is None:
            QtWidgets.QMessageBox.warning(parent or self, "No data", "Load grid first.")
            return
#        self.grid = np.flipud(self.grid)
        self.grid = np.fliplr(self.grid)
        self.grid = -self.grid
        self.update_image()

    def fill_holes(self, parent=None):
        if self.grid is None:
            QtWidgets.QMessageBox.warning(parent or self, "No data", "Load grid first.")
            return

        tst = np.isnan(self.grid)
        for (iy, ix) in self.seed_points:
            filled = flood(tst, seed_point=(iy, ix))
            tst[filled] = False

        print("grid.shape:", self.grid.shape)
        print("xi len:", len(self.xi))
        print("yi len:", len(self.yi))

        grid_x, grid_y = np.meshgrid(self.xi, self.yi)

        print("grid_x.shape:", grid_x.shape)
        print("grid_y.shape:", grid_y.shape)
        print("tst.shape:", tst.shape)

        interp_points = np.column_stack((grid_x[tst], grid_y[tst]))

        valid = ~np.isnan(self.grid)
        interp_values = griddata(
            (grid_x[valid], grid_y[valid]),
            self.grid[valid],
            interp_points,
            method='nearest'
        )

        self.grid[tst] = interp_values
        self.update_image()

    def get_zero_point_value(self, x, y):
        s = self.zero_window_size // 2
        grid = self.grid
        h, w = grid.shape
        xmin = max(0, x - s)
        xmax = min(w, x + s + 1)
        ymin = max(0, y - s)
        ymax = min(h, y + s + 1)
        window = grid[ymin:ymax, xmin:xmax]

        # Wartości bez NaN
        vals = window[~np.isnan(window)]
        if len(vals) == 0:
            return np.nan

        # Odrzuć odstające (np. 2 sigma od mediany)
        median = np.median(vals)
        std = np.std(vals)
        non_outliers = vals[np.abs(vals - median) < self.zero_sigma * std]

        # Jeżeli po odrzuceniu nie ma wartości – bierz medianę
        if len(non_outliers) == 0:
            return median
        return np.mean(non_outliers)

    def fit_plane_to_grid(self, grid, x, y, s=100):
        h, w = grid.shape
        xmin = max(0, x - s)
        xmax = min(w, x + s + 1)
        ymin = max(0, y - s)
        ymax = min(h, y + s + 1)

        window = grid[ymin:ymax, xmin:xmax]

        yy, xx = np.mgrid[ymin:ymax, xmin:xmax]
        zz = window

        # Zamień na 1D i odrzuć NaN
        X = xx.flatten()
        Y = yy.flatten()
        Z = zz.flatten()
        mask = ~np.isnan(Z)
        X = X[mask]
        Y = Y[mask]
        Z = Z[mask]

        if len(Z) < 10:
            raise ValueError("Zbyt mało ważnych danych do dopasowania płaszczyzny")

        A = np.vstack((X, Y)).T
        from sklearn.linear_model import LinearRegression
        model = LinearRegression().fit(A, Z)
        a, b = model.coef_
        c = model.intercept_

        return a, b, c


    def fit_plane_to_grid_robust(self, grid, x, y, s=100):
        h, w = grid.shape
        xmin = max(0, x - s)
        xmax = min(w, x + s + 1)
        ymin = max(0, y - s)
        ymax = min(h, y + s + 1)

        window = grid[ymin:ymax, xmin:xmax]
        yy, xx = np.mgrid[ymin:ymax, xmin:xmax]
        zz = window

        X = xx.flatten()
        Y = yy.flatten()
        Z = zz.flatten()
        mask = ~np.isnan(Z)
        X = X[mask]
        Y = Y[mask]
        Z = Z[mask]

        if len(Z) < 10:
            raise ValueError("Zbyt mało ważnych danych do dopasowania płaszczyzny")

        A = np.vstack((X, Y)).T

        # Tu używamy RANSAC, żeby być odpornym na odstające wartości
        from sklearn.linear_model import RANSACRegressor, LinearRegression
        base_model = LinearRegression()
        model = RANSACRegressor(base_model, min_samples=100, residual_threshold=200.0, random_state=42)
        model.fit(A, Z)
        a, b = model.estimator_.coef_
        c = model.estimator_.intercept_

        return a, b, c

    def fit_plane_to_grid_median_filter(self, grid, x, y, s=100, outlier_thresh=300.0):
        h, w = grid.shape
        xmin = max(0, x - s)
        xmax = min(w, x + s + 1)
        ymin = max(0, y - s)
        ymax = min(h, y + s + 1)

        window = grid[ymin:ymax, xmin:xmax]
        yy, xx = np.mgrid[ymin:ymax, xmin:xmax]
        zz = window

        X = xx.flatten()
        Y = yy.flatten()
        Z = zz.flatten()
        mask = ~np.isnan(Z)
        X = X[mask]
        Y = Y[mask]
        Z = Z[mask]

        if len(Z) < 10:
            raise ValueError("Zbyt mało ważnych danych do dopasowania płaszczyzny")

        # Usuwanie wartości odstających na podstawie mediany
        z_median = np.median(Z)
        mad = np.median(np.abs(Z - z_median))  # Median Absolute Deviation
        # Pozostaw tylko te punkty, które nie są bardzo daleko od mediany
        robust_mask = np.abs(Z - z_median) < outlier_thresh * mad
        X = X[robust_mask]
        Y = Y[robust_mask]
        Z = Z[robust_mask]

        if len(Z) < 10:
            raise ValueError("Za mało danych po odrzuceniu outlierów")

        A = np.vstack((X, Y)).T
        from sklearn.linear_model import LinearRegression
        model = LinearRegression().fit(A, Z)
        a, b = model.coef_
        c = model.intercept_

        return a, b, c


    def mouse_clicked(self, event):
        if self.grid is None:
            return
        vb = self.image_view.getView()
        mouse_point = vb.mapSceneToView(event.scenePos())
        x = int(round(mouse_point.x()))
        y = int(round(mouse_point.y()))
        if 0 <= x < self.grid.shape[1] and 0 <= y < self.grid.shape[0]:
            if self.zero_point_mode:
                # value = self.grid[y, x]
                value = self.get_zero_point_value(x, y)
                if np.isnan(value):
                    QtWidgets.QMessageBox.warning(self, "Brak danych", "Wybrany punkt nie zawiera wartości (NaN).")
                    self.zero_point_mode = False
                    return
                # Przesuwamy cały skan w osi Z
                self.grid = self.grid - value
                self.update_image()
                self.zero_point_mode = False
                # QtWidgets.QMessageBox.information(self, "Sukces", f"Ustawiono nowy punkt zerowy na ({x},{y}) o wysokości {value:.2f}.")

                min_val = self.hist_min_line.value()-value
                max_val = self.hist_max_line.value()-value

                self.update_histogram()
                self.hist_min_line.setValue(min_val)
                self.hist_max_line.setValue(max_val)
                return
            
            elif self.tilt_mode:
                self.tilt_mode = False
                # a, b, c = self.fit_plane_to_grid_robust(self.grid, x, y, s=100)
                a, b, c = self.fit_plane_to_grid_median_filter(self.grid, x, y, s=500, outlier_thresh=300.0)
                # Możesz teraz utworzyć macierz tej samej wielkości co grid:
                rows, cols = self.grid.shape
                yy, xx = np.mgrid[0:rows, 0:cols]
                plane = a * xx + b * yy + c

                # Korekta:
                self.grid = self.grid + plane
                self.update_image()
                self.update_histogram()
                return

            if event.modifiers() & QtCore.Qt.ShiftModifier:
                self.seed_points.append((y, x))
                scatter = pg.ScatterPlotItem([x], [y], size=10, brush=pg.mkBrush('r'))
                vb.addItem(scatter)

    def toggle_colormap(self):
        self.is_colormap = not self.is_colormap
        self.update_image()


#    @measure_time
    def update_image(self, vmin=None, vmax=None):
        if self.grid is not None:
            
            if vmin is None or vmax is None:
                if hasattr(self, 'hist_min_line') and hasattr(self, 'hist_max_line'):
                    vmin = min(self.hist_min_line.value(), self.hist_max_line.value())
                    vmax = max(self.hist_min_line.value(), self.hist_max_line.value())
                else:
                    vmin = float(np.min(self.grid))
                    vmax = float(np.max(self.grid))

            data = self.grid.T
            self.masked = data.copy()
            self.masked[(self.masked < vmin) | (self.masked > vmax)] = np.nan
            if np.isnan(self.masked).all():
                self.masked = np.zeros_like(self.masked)
            image_item = self.image_view.getImageItem()
            if self.is_colormap:
                lut = pg.colormap.get('turbo').getLookupTable(0.0, 1.0, 256)
                image_item.setLookupTable(lut)
            else:
                image_item.setLookupTable(None)
            self.image_view.setImage(self.masked, autoLevels=True, autoRange=False)
        self.seed_points = []

    def update_image2(self, vmin=None, vmax=None):
        if self.grid is not None:
            if vmin is None or vmax is None:
                if hasattr(self, 'hist_min_line') and hasattr(self, 'hist_max_line'):
                    vmin = min(self.hist_min_line.value(), self.hist_max_line.value())
                    vmax = max(self.hist_min_line.value(), self.hist_max_line.value())
                else:
                    vmin = float(np.min(self.grid))
                    vmax = float(np.max(self.grid))

            data = self.grid.T
            self.masked = data.copy()
            self.masked[(self.masked < vmin) | (self.masked > vmax)] = np.nan

            # Specjalna wartość powyżej vmax
            special = vmax + max(1.0, (vmax - vmin) * 0.01)
            masked_for_vis = self.masked.copy()
            masked_for_vis[np.isnan(masked_for_vis)] = special

            image_item = self.image_view.getImageItem()
            if self.is_colormap:
                lut = pg.colormap.get('turbo').getLookupTable(0.0, 1.0, 256)
                lut = np.vstack([lut, [255, 0, 0, 255]])  # czerwony na końcu
                image_item.setLookupTable(lut)
                # Kluczowe: levels od vmin do special
                self.image_view.setImage(masked_for_vis, autoLevels=False, levels=(vmin, special), autoRange=False)
            else:
                # W wersji gray nie podświetlisz, ale możesz np. zrobić specjalny kolor: 255
                masked_for_vis_gray = self.masked.copy()
                masked_for_vis_gray[np.isnan(masked_for_vis_gray)] = 255
                self.image_view.setImage(masked_for_vis_gray, autoLevels=True, autoRange=False)

        self.seed_points = []
