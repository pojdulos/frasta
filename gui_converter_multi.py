import sys
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from skimage.segmentation import flood
from scipy.interpolate import griddata
import trimesh

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
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.image_view)
        self.setLayout(layout)

        self.seed_points = []
        self.grid = None
        self.xi = None
        self.yi = None
        self.px_x = None
        self.px_y = None
        self.orig_data = None

        self.image_view.getView().scene().sigMouseClicked.connect(self.mouse_clicked)

    def set_data(self, grid, xi, yi, px_x, px_y, x, y, z):
        self.grid = grid
        self.xi = xi
        self.yi = yi
        self.px_x = px_x
        self.px_y = px_y
        self.orig_data = (x, y, z)
        self.update_image()

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

    def save_as_mesh(self, grid, px_x=1.38, px_y=1.38):
        v, f = grid_to_mesh_vectorized(grid, px_x, px_y)
        mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
        mesh.export("mesh_output.obj")


    def save_file(self, parent=None):
        if self.grid is None:
            return
        fname, nn = QtWidgets.QFileDialog.getSaveFileName(parent or self, "Save as...", "", "NPZ (*.npz);OBJ (*.obj)")
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
        self.grid = np.flipud(self.grid)
        self.grid = -self.grid
        self.update_image()

    def fill_holes(self, parent=None):
        if self.grid is None or not self.seed_points:
            QtWidgets.QMessageBox.warning(parent or self, "No data", "Load grid and select at least one seed point.")
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

    def mouse_clicked(self, event):
        if self.grid is None:
            return
        vb = self.image_view.getView()
        mouse_point = vb.mapSceneToView(event.scenePos())
        x = int(round(mouse_point.x()))
        y = int(round(mouse_point.y()))
        if 0 <= x < self.grid.shape[1] and 0 <= y < self.grid.shape[0]:
            self.seed_points.append((y, x))
            scatter = pg.ScatterPlotItem([x], [y], size=10, brush=pg.mkBrush('r'))
            vb.addItem(scatter)
        # Jeśli klik poza gridem, ignoruj

    def update_image(self):
        if self.grid is not None:
            self.image_view.setImage(self.grid.T, autoLevels=True)
        self.seed_points = []

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scan Loader & Hole Filler (Multi-Tab)")

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

        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        file_menu.addAction(open_action)
        file_menu.addAction(save_action)
        actions_menu = menubar.addMenu("&Actions")
        actions_menu.addAction(fill_action)
        actions_menu.addAction(flip_action)

        self.worker = None
        self.thread = None

    def current_tab(self):
        tab = self.tabs.currentWidget()
        return tab

    def open_file(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open file", "", "CSV, NPY, NPZ (*.csv *.npy *.npz)")
        if not fname:
            return
        tab = ScanTab()
        self.tabs.addTab(tab, fname.split('/')[-1])
        self.tabs.setCurrentWidget(tab)
        if fname.endswith('.csv'):
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
        elif fname.endswith('.npz'):
            data = np.load(fname)
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

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
