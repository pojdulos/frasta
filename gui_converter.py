import sys
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from skimage.segmentation import flood
from scipy.interpolate import griddata
import trimesh
from PyQt5.QtGui import QIcon

from overlayViewer import OverlayViewer
from aboutDialog import AboutDialog

def grid_to_mesh_vectorized(grid, pixel_size_x=1.0, pixel_size_y=1.0):
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



class GridWorker(QtCore.QObject):
    progress = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal(object, object, object, float, float, object, object, object)  # grid, xi, yi, px_x, px_y, x, y, z

    def __init__(self, fname):
        super().__init__()
        self.fname = fname

    @QtCore.pyqtSlot()
    def process(self):
        chunk_size = 100_000
        total = sum(1 for _ in open(self.fname, encoding="utf-8"))
        chunks = []
        for i, chunk in enumerate(pd.read_csv(self.fname, sep=';', header=None, names=['x','y','z'], chunksize=chunk_size)):
            chunks.append(chunk)
            self.progress.emit(int(20 + 30 * (i*chunk_size/total)))
        df = pd.concat(chunks, ignore_index=True)
        x, y, z = df['x'].values, df['y'].values, df['z'].values

        dx = np.diff(np.sort(np.unique(x)))
        dy = np.diff(np.sort(np.unique(y)))
        px_x = np.median(dx[dx > 0]).round(2)
        px_y = np.median(dy[dy > 0]).round(2)
        px_x, px_y = 1.38, 1.38

        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        grid_size_x = int((x_max - x_min) / px_x) + 1
        grid_size_y = int((y_max - y_min) / px_y) + 1

        grid = np.full((grid_size_y, grid_size_x), np.nan, dtype=np.float32)
        counts = np.zeros_like(grid, dtype=np.int32)
        N = len(x)
        for idx, (xi, yi, zi) in enumerate(zip(x, y, z)):
            ix = int(round((xi - x_min) / px_x))
            iy = int(round((yi - y_min) / px_y))
            if 0 <= ix < grid_size_x and 0 <= iy < grid_size_y:
                if np.isnan(grid[iy, ix]):
                    grid[iy, ix] = zi
                else:
                    grid[iy, ix] += zi
                counts[iy, ix] += 1
            if idx % max(1, N//50) == 0:
                self.progress.emit(50 + int(49 * idx / N))

        mask_dup = (counts > 1)
        grid[mask_dup] = grid[mask_dup] / counts[mask_dup]
        xi_grid = np.linspace(x_min, x_max, grid_size_x)
        yi_grid = np.linspace(y_min, y_max, grid_size_y)
        self.progress.emit(100)
        self.finished.emit(grid, xi_grid, yi_grid, px_x, px_y, x, y, z)

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

        # self.hist_min_line = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('b', width=2))
        # self.hist_max_line = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('r', width=2))

        # self.hist_min_line.sigPositionChanged.connect(self.on_hist_line_changed)
        # self.hist_max_line.sigPositionChanged.connect(self.on_hist_line_changed)

        # self.hist_widget.addItem(self.hist_min_line)
        # self.hist_widget.addItem(self.hist_max_line)

        hlayout = QtWidgets.QVBoxLayout(self)
        hlayout.addWidget(self.image_view, stretch=1)
        hlayout.addWidget(self.hist_widget)


        # self.range_min_sb = QtWidgets.QDoubleSpinBox()
        # self.range_max_sb = QtWidgets.QDoubleSpinBox()
        # self.range_min_sb.setDecimals(2)
        # self.range_max_sb.setDecimals(2)
        # self.range_min_sb.setMaximum(1e9)
        # self.range_max_sb.setMaximum(1e9)
        # self.range_min_sb.setMinimum(-1e9)
        # self.range_max_sb.setMinimum(-1e9)
        # self.range_min_sb.setPrefix('Min: ')
        # self.range_max_sb.setPrefix('Max: ')

        # self.range_min_sb.valueChanged.connect(self.on_hist_range)
        # self.range_max_sb.valueChanged.connect(self.on_hist_range)

        # sb_layout = QtWidgets.QVBoxLayout()
        # sb_layout.addWidget(self.range_min_sb)
        # sb_layout.addWidget(self.range_max_sb)
        # hlayout.addLayout(sb_layout)

        self.setLayout(hlayout)


        # layout = QtWidgets.QVBoxLayout(self)
        # layout.addWidget(self.image_view)
        # self.setLayout(layout)

        self.zero_point_mode = False

        self.seed_points = []
        self.grid = None
        self.xi = None
        self.yi = None
        self.px_x = None
        self.px_y = None
        self.orig_data = None

        self.is_colormap = False
        self.current_colormap = 'gray'  # lub None

        self.zero_window_size = 7  # lub inna liczba nieparzysta
        self.zero_sigma = 2.0      # ile odchyleń przyjmujesz jako "nie odstające"

        self.image_view.getView().scene().sigMouseClicked.connect(self.mouse_clicked)

    def on_hist_line_changed(self):
        vmin = min(self.hist_min_line.value(), self.hist_max_line.value())
        vmax = max(self.hist_min_line.value(), self.hist_max_line.value())
        self.update_image(vmin, vmax)

    # def on_hist_range(self):
    #     self.update_image

    # def update_histogram(self):
    #     data = self.grid[~np.isnan(self.grid)]
    #     y, x = np.histogram(data, bins=512)
    #     self.hist_widget.clear()
    #     self.hist_plot = self.hist_widget.plot(x, y, stepMode=True, fillLevel=0, brush=(150, 150, 150, 150))
    #     if data.size > 0:
    #         vmin = float(np.min(data))
    #         vmax = float(np.max(data))

    #         if not hasattr(self, 'hist_min_line'):
    #             self.hist_min_line = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('b', width=2))
    #             self.hist_widget.addItem(self.hist_min_line)
    #             self.hist_min_line.setValue(vmin)
    #             self.hist_min_line.sigPositionChanged.connect(self.on_hist_line_changed)
    #         if not hasattr(self, 'hist_max_line'):
    #             self.hist_max_line = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('r', width=2))
    #             self.hist_widget.addItem(self.hist_max_line)
    #             self.hist_max_line.setValue(vmax)
    #             self.hist_max_line.sigPositionChanged.connect(self.on_hist_line_changed)

    #         vmin = float(max(np.min(data),self.hist_min_line.value()))
    #         vmax = float(min(np.max(data),self.hist_max_line.value()))

    #         print(f"vmin: {vmin}, vmax: {vmax}")

    #         self.hist_min_line.setValue(vmin)
    #         self.hist_max_line.setValue(vmax)

    def update_histogram(self):
        if self.grid is None:
            return
        data = self.grid[~np.isnan(self.grid)]
        if data.size == 0:
            self.hist_widget.clear()
            return

        y, x = np.histogram(data, bins=512)
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

        # Utwórz nowe linie
        self.hist_min_line = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('b', width=2))
        self.hist_max_line = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('r', width=2))
        self.hist_widget.addItem(self.hist_min_line)
        self.hist_widget.addItem(self.hist_max_line)
        self.hist_min_line.setValue(min_val)
        self.hist_max_line.setValue(max_val)
        self.hist_min_line.sigPositionChanged.connect(self.on_hist_line_changed)
        self.hist_max_line.sigPositionChanged.connect(self.on_hist_line_changed)



    def set_zero_point_mode(self):
        self.zero_point_mode = True
        # QtWidgets.QMessageBox.information(self, "Wybierz punkt", "Kliknij na widoku skanu punkt, który ma być nowym zerem.")

    def set_data(self, grid, xi, yi, px_x, px_y, x, y, z):
        self.grid = grid
        self.xi = xi
        self.yi = yi
        self.px_x = px_x
        self.px_y = px_y
        self.orig_data = (x, y, z)
        self.update_image()
        self.update_histogram()

    def set_data_npz(self, data):
        self.grid = data['grid']
        self.xi = data['xi']
        self.yi = data['yi']
        self.px_x = data['px_x']
        self.px_y = data['px_y']
        if 'x_data' in data:
            self.orig_data = (data['x_data'], data['y_data'], data['z_data'])
        else:
            self.orig_data = None
        self.update_image()
        self.update_histogram()

    def save_as_mesh(self, grid, px_x=1.38, px_y=1.38):
        v, f = grid_to_mesh_vectorized(grid, px_x, px_y)
        mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
        mesh.export("mesh_output.obj")


    def save_file(self, parent=None):
        if self.grid is None:
            return
        fname, nn = QtWidgets.QFileDialog.getSaveFileName(parent or self, "Save as...", "", "NPZ (*.npz)")
        print(nn)
        if fname:
            if fname.endswith(".npz"):
                to_save = dict(grid=self.grid, xi=self.xi, yi=self.yi, px_x=self.px_x, px_y=self.px_y)
                if self.orig_data:
                    x, y, z = self.orig_data
                    to_save.update(x_data=x, y_data=y, z_data=z)
                np.savez(fname, **to_save)
            elif fname.endswith(".obj"):
                self.save_as_mesh(self.grid)

    def flip_scan(self, parent=None):
        if self.grid is None:
            QtWidgets.QMessageBox.warning(parent or self, "No data", "Load grid first.")
            return
#        self.grid = np.flipud(self.grid)
        self.grid = np.fliplr(self.grid)
        self.grid = -self.grid
        self.update_image()

    def fill_holes(self, parent=None):
        if self.grid is None: #or not self.seed_points:
            QtWidgets.QMessageBox.warning(parent or self, "No data", "Load grid first.")
            return

        tst = np.isnan(self.grid)
        for (iy, ix) in self.seed_points:
            filled = flood(tst, seed_point=(iy, ix))
            tst[filled] = False

        grid_x, grid_y = np.meshgrid(self.xi, self.yi)
        interp_points = np.column_stack((grid_x[tst], grid_y[tst]))

        if self.orig_data:
            x, y, z = self.orig_data
            interp_values = griddata((x, y), z, interp_points, method='nearest')
        else:
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
            
            if event.modifiers() & QtCore.Qt.ShiftModifier:
                self.seed_points.append((y, x))
                scatter = pg.ScatterPlotItem([x], [y], size=10, brush=pg.mkBrush('r'))
                vb.addItem(scatter)

    def toggle_colormap(self):
        self.is_colormap = not self.is_colormap
        self.update_image()

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
            masked = data.copy()
            masked[(masked < vmin) | (masked > vmax)] = np.nan
            if np.isnan(masked).all():
                masked = np.zeros_like(masked)
            image_item = self.image_view.getImageItem()
            if self.is_colormap:
                lut = pg.colormap.get('turbo').getLookupTable(0.0, 1.0, 256)
                image_item.setLookupTable(lut)
            else:
                image_item.setLookupTable(None)
            self.image_view.setImage(masked, autoLevels=True)
        self.seed_points = []


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scan Loader & Hole Filler (Multi-Tab)")
        self.setGeometry(100, 100, 1000, 600)

        self.recent_files = []
        self.max_recent_files = 10
        self.settings = QtCore.QSettings("IITiS PAN", "FRASTA - converter")
        self.load_recent_files()

        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        open_action = QtWidgets.QAction("Open...", self)
        open_action.triggered.connect(self.open_file)
        save_action = QtWidgets.QAction("Save as...", self)
        save_action.triggered.connect(self.save_file)
        fill_action = QtWidgets.QAction("Fill holes", self)
        fill_action.triggered.connect(self.fill_holes)
        flip_action = QtWidgets.QAction("Flip & reverse", self)
        flip_action.triggered.connect(self.flip_scan)
        compare_action = QtWidgets.QAction("Porównaj skany...", self)
        compare_action.triggered.connect(self.compare_scans)
        zero_action = QtWidgets.QAction("Ustaw punkt zerowy", self)
        zero_action.triggered.connect(self.set_zero_point_mode)

        about_action = QtWidgets.QAction("About...", self)
        about_action.triggered.connect(self.show_about_dialog)

        exit_action = QtWidgets.QAction("Exit", self)
        exit_action.triggered.connect(self.close)

        colormap_action = QtWidgets.QAction("Toggle colormap", self)
        colormap_action.setCheckable(True)
        colormap_action.setChecked(False)
        colormap_action.triggered.connect(self.toggle_colormap_current_tab)

        # open_action.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton))
        # save_action.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton))

        open_action.setIcon(QIcon("icons/icons8-open-file-50.png"))
        save_action.setIcon(QIcon("icons/icons8-save-50.png"))
        fill_action.setIcon(QIcon("icons/icons8-fill-color-50.png"))
        flip_action.setIcon(QIcon("icons/icons8-flip-48.png"))
        compare_action.setIcon(QIcon("icons/icons8-compare-50.png"))
        zero_action.setIcon(QIcon("icons/icons8-eyedropper-50.png"))
        about_action.setIcon(QIcon("icons/icons8-about-50.png"))
        exit_action.setIcon(QIcon("icons/icons8-exit-50.png"))


        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        file_menu.addAction(open_action)
        file_menu.addAction(save_action)

        self.recent_menu = QtWidgets.QMenu("Recent files", self)
        file_menu.addMenu(self.recent_menu)
        self.update_recent_files_menu()

        file_menu.addAction(exit_action)
        


        actions_menu = menubar.addMenu("&Actions")
        actions_menu.addAction(fill_action)
        actions_menu.addAction(flip_action)
        actions_menu.addAction(compare_action)
        actions_menu.addAction(zero_action)

        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction(about_action)

        self.toolbar = self.addToolBar("Tools")
        self.toolbar.addAction(open_action)
        self.toolbar.addAction(save_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(fill_action)
        self.toolbar.addAction(flip_action)
        self.toolbar.addAction(zero_action)
        self.toolbar.addAction(colormap_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(compare_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(about_action)
        self.toolbar.addAction(exit_action)

        self.worker = None
        self.thread = None

    def toggle_colormap_current_tab(self):
        tab = self.current_tab()
        if tab:
            tab.toggle_colormap()

    def set_zero_point_mode(self):
        tab = self.current_tab()
        if tab:
            tab.set_zero_point_mode()

    def show_about_dialog(self):
        print("About")
        dlg = AboutDialog(self)
        dlg.exec_()

    def closeEvent(self, event):
        self.settings.setValue("recentFiles", self.recent_files)
        event.accept()

    def add_to_recent_files(self, path):
        if path in self.recent_files:
            self.recent_files.remove(path)
        self.recent_files.insert(0, path)
        if len(self.recent_files) > self.max_recent_files:
            self.recent_files = self.recent_files[:self.max_recent_files]
        self.update_recent_files_menu()
        self.settings.setValue("recentFiles", self.recent_files)

    def load_recent_files(self):
        self.recent_files = self.settings.value("recentFiles", [], type=list)
        self.max_recent_files = 10

    def update_recent_files_menu(self):
        self.recent_menu.clear()
        if not self.recent_files:
            action = QtWidgets.QAction("No recent files", self)
            action.setEnabled(False)
            self.recent_menu.addAction(action)
            return
        for path in self.recent_files:
            action = QtWidgets.QAction(path, self)
            action.triggered.connect(lambda checked, p=path: self.open_file_from_recent(p))
            self.recent_menu.addAction(action)

    def current_tab(self):
        tab = self.tabs.currentWidget()
        return tab

    def open_file(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open file", "", "CSV, NPY, NPZ (*.csv *.dat *.npy *.npz)")
        if not fname:
            return
        tab = ScanTab()
        self.tabs.addTab(tab, fname.split('/')[-1])
        self.tabs.setCurrentWidget(tab)
        if fname.endswith('.csv') or fname.endswith('.dat'):
            dlg = QtWidgets.QProgressDialog("Wczytywanie i gridowanie...", None, 0, 100, self)
            dlg.setWindowModality(QtCore.Qt.ApplicationModal)
            dlg.setAutoClose(True)
            dlg.setCancelButton(None)
            dlg.setValue(0)
            self.worker = GridWorker(fname)
            self.thread = QtCore.QThread()
            self.worker.moveToThread(self.thread)
            self.worker.progress.connect(dlg.setValue)
            self.worker.finished.connect(lambda *args: tab.set_data(*args))
            self.worker.finished.connect(self.thread.quit)
            self.thread.started.connect(self.worker.process)
            self.thread.start()
            dlg.exec_()
            self.add_to_recent_files(fname)
        elif fname.endswith('.npz'):
            data = np.load(fname)
            if 'grid' in data:
                tab.set_data_npz(data)
                self.add_to_recent_files(fname)
            else:
                QtWidgets.QMessageBox.warning(self, "Format error", "NPZ does not contain a grid.")
                self.tabs.removeTab(self.tabs.indexOf(tab))
                return
        else:
            QtWidgets.QMessageBox.warning(self, "Unknown format", "Unsupported file type.")
            self.tabs.removeTab(self.tabs.indexOf(tab))
            return

    def open_file_from_recent(self, path):
        if not QtCore.QFile.exists(path):
            QtWidgets.QMessageBox.warning(self, "File not found", f"File not found:\n{path}")
            self.recent_files.remove(path)
            self.update_recent_files_menu()
            return
        # ...prawie to samo co w open_file, ale bez dialogu...
        tab = ScanTab()
        self.tabs.addTab(tab, path.split('/')[-1])
        self.tabs.setCurrentWidget(tab)
        if path.endswith('.csv') or path.endswith('.dat'):
            # ... kod z wątkiem i gridowaniem jak w open_file ...
            dlg = QtWidgets.QProgressDialog("Wczytywanie i gridowanie...", None, 0, 100, self)
            dlg.setWindowModality(QtCore.Qt.ApplicationModal)
            dlg.setAutoClose(True)
            dlg.setCancelButton(None)
            dlg.setValue(0)
            self.worker = GridWorker(path)
            self.thread = QtCore.QThread()
            self.worker.moveToThread(self.thread)
            self.worker.progress.connect(dlg.setValue)
            self.worker.finished.connect(lambda *args: tab.set_data(*args))
            self.worker.finished.connect(self.thread.quit)
            self.thread.started.connect(self.worker.process)
            self.thread.start()
            dlg.exec_()
        elif path.endswith('.npz'):
            data = np.load(path)
            if 'grid' in data:
                tab.set_data_npz(data)
            else:
                QtWidgets.QMessageBox.warning(self, "Format error", "NPZ does not contain a grid.")
                self.tabs.removeTab(self.tabs.indexOf(tab))
                return
        else:
            QtWidgets.QMessageBox.warning(self, "Unknown format", "Unsupported file type.")
            self.tabs.removeTab(self.tabs.indexOf(tab))
            return
        self.add_to_recent_files(path)


    def save_file(self):
        tab = self.current_tab()
        if tab:
            tab.save_file(self)

    def flip_scan(self):
        tab = self.current_tab()
        if tab:
            tab.flip_scan(self)

    def fill_holes(self):
        tab = self.current_tab()
        if tab:
            tab.fill_holes(self)

    def compare_scans(self):
        if self.tabs.count() < 2:
            QtWidgets.QMessageBox.warning(self, "Za mało skanów", "Musisz mieć przynajmniej 2 skany!")
            return

        # Dialog wyboru zakładek
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Wybierz skany do porównania")
        layout = QtWidgets.QVBoxLayout(dialog)
        label1 = QtWidgets.QLabel("Referencyjny skan:")
        label2 = QtWidgets.QLabel("Skan do dopasowania:")
        cb1 = QtWidgets.QComboBox()
        cb2 = QtWidgets.QComboBox()
        names = [self.tabs.tabText(i) for i in range(self.tabs.count())]
        cb1.addItems(names)
        cb2.addItems(names)
        ok_btn = QtWidgets.QPushButton("OK")
        cancel_btn = QtWidgets.QPushButton("Anuluj")
        hl = QtWidgets.QHBoxLayout()
        hl.addWidget(ok_btn)
        hl.addWidget(cancel_btn)
        layout.addWidget(label1)
        layout.addWidget(cb1)
        layout.addWidget(label2)
        layout.addWidget(cb2)
        layout.addLayout(hl)

        def accept():
            if cb1.currentIndex() == cb2.currentIndex():
                QtWidgets.QMessageBox.warning(dialog, "Błąd", "Wybierz dwa różne skany!")
                return
            dialog.accept()
        ok_btn.clicked.connect(accept)
        cancel_btn.clicked.connect(dialog.reject)

        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return

        idx1 = cb1.currentIndex()
        idx2 = cb2.currentIndex()
        tab1 = self.tabs.widget(idx1)
        tab2 = self.tabs.widget(idx2)

        # Teraz pobieramy gridy i przekazujemy do narzędzia różnicowego
        grid1 = tab1.grid
        grid2 = tab2.grid

        # --- Tu otwieramy okno narzędzia różnicowego ---
        self.viewer = OverlayViewer(grid1, grid2)
        self.viewer.setWindowTitle(f"Porównanie: {names[idx1]} vs {names[idx2]}")
        self.viewer.show()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
