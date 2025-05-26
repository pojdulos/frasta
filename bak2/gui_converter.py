import sys
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from skimage.segmentation import flood
from scipy.interpolate import griddata

from PyQt5 import QtCore

class CsvLoader(QtCore.QObject):
    progress = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal(object)  # DataFrame

    def __init__(self, fname):
        super().__init__()
        self.fname = fname

    @QtCore.pyqtSlot()
    def load(self):
        chunks = []
        chunk_size = 10_000  # np. 10k wierszy
        total = sum(1 for _ in open(self.fname, encoding="utf-8"))  # liczymy wiersze
        for i, chunk in enumerate(pd.read_csv(self.fname, sep=';', header=None, names=['x','y','z'], chunksize=chunk_size)):
            chunks.append(chunk)
            self.progress.emit(min(100, int(100*(i*chunk_size)/total)))
        df = pd.concat(chunks, ignore_index=True)
        self.progress.emit(100)
        self.finished.emit(df)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scan Loader & Hole Filler")

        # Central widget
        self.image_view = pg.ImageView()
        self.setCentralWidget(self.image_view)

        # Seed points
        self.seed_points = []

        # Menu
        open_action = QtWidgets.QAction("Open...", self)
        open_action.triggered.connect(self.open_file)
        save_action = QtWidgets.QAction("Save as...", self)
        save_action.triggered.connect(self.save_file)
        fill_action = QtWidgets.QAction("Fill holes", self)
        fill_action.triggered.connect(self.fill_holes)

        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        file_menu.addAction(open_action)
        file_menu.addAction(save_action)
        actions_menu = menubar.addMenu("&Actions")
        actions_menu.addAction(fill_action)

        # Data holders
        self.grid = None
        self.xi = None
        self.yi = None
        self.px_x = None
        self.px_y = None
        self.orig_data = None  # for scatter source points

        # Mouse click to add seed points
        self.image_view.getView().scene().sigMouseClicked.connect(self.mouse_clicked)

    def csv_loaded(self, df):
        # dalej tak jak było: gridowanie, itd.
        x, y, z = df['x'].values, df['y'].values, df['z'].values
        # Create grid as in your conversion script
        dx = np.diff(np.sort(np.unique(x)))
        dy = np.diff(np.sort(np.unique(y)))
        px_x = np.median(dx[dx > 0]).round(2)
        px_y = np.median(dy[dy > 0]).round(2)
        px_x, px_y = 1.38, 1.38  # hardcoded, as in your script
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        grid_size_x = int((x_max - x_min) / px_x) + 1
        grid_size_y = int((y_max - y_min) / px_y) + 1
        grid = np.full((grid_size_y, grid_size_x), np.nan, dtype=np.float32)
        counts = np.zeros_like(grid, dtype=np.int32)
        for xi, yi, zi in zip(x, y, z):
            ix = int(round((xi - x_min) / px_x))
            iy = int(round((yi - y_min) / px_y))
            if 0 <= ix < grid_size_x and 0 <= iy < grid_size_y:
                if np.isnan(grid[iy, ix]):
                    grid[iy, ix] = zi
                else:
                    grid[iy, ix] += zi
                counts[iy, ix] += 1
        mask_dup = (counts > 1)
        grid[mask_dup] = grid[mask_dup] / counts[mask_dup]
        xi_grid = np.linspace(x_min, x_max, grid_size_x)
        yi_grid = np.linspace(y_min, y_max, grid_size_y)
        self.grid = grid
        self.xi = xi_grid
        self.yi = yi_grid
        self.px_x = px_x
        self.px_y = px_y
        self.orig_data = (x, y, z)

    def open_file(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open file", "", "CSV, NPY, NPZ (*.csv *.npy *.npz)")
        if not fname:
            return
        if fname.endswith('.csv'):
            # Tworzymy modalny progress
            dlg = QtWidgets.QProgressDialog("Wczytywanie pliku...", None, 0, 100, self)
            dlg.setWindowModality(QtCore.Qt.ApplicationModal)
            dlg.setAutoClose(True)
            dlg.setCancelButton(None)
            dlg.setValue(0)
            # Worker i wątek
            self.csv_loader = CsvLoader(fname)
            self.thread = QtCore.QThread()
            self.csv_loader.moveToThread(self.thread)
            self.csv_loader.progress.connect(dlg.setValue)
            self.csv_loader.finished.connect(self.csv_loaded)
            self.csv_loader.finished.connect(self.thread.quit)
            self.thread.started.connect(self.csv_loader.load)
            self.thread.start()
            dlg.exec_()
        elif fname.endswith('.npz'):
            data = np.load(fname)
            # Guess format
            if 'grid' in data:
                self.grid = data['grid']
                self.xi = data['xi']
                self.yi = data['yi']
                self.px_x = data['px_x']
                self.px_y = data['px_y']
            else:
                QtWidgets.QMessageBox.warning(self, "Format error", "NPZ does not contain a grid.")
                return
        else:
            QtWidgets.QMessageBox.warning(self, "Unknown format", "Unsupported file type.")
            return
        self.update_image()

    def save_file(self):
        if self.grid is None:
            return
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save as...", "", "NPZ (*.npz)")
        if fname:
            np.savez(fname, grid=self.grid, xi=self.xi, yi=self.yi, px_x=self.px_x, px_y=self.px_y)

    def fill_holes(self):
        if self.grid is None or not self.seed_points:
            QtWidgets.QMessageBox.warning(self, "No data", "Load grid and select at least one seed point.")
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
            # fallback: interpolate from grid itself
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
        # Mamy x,y w przestrzeni "świata", musimy zamienić na indeksy obrazu
        # self.xi: długość grid.shape[1], self.yi: grid.shape[0]
        x, y = mouse_point.x(), mouse_point.y()
        ix = np.abs(self.xi - x).argmin()
        iy = np.abs(self.yi - y).argmin()
        # UWAGA: Dla wyświetlania .T (transpozycja), należy zamienić miejscami ix/iy!
        self.seed_points.append((iy, ix))
        scatter = pg.ScatterPlotItem([ix], [iy], size=10, brush=pg.mkBrush('r'))
        self.image_view.addItem(scatter)
        print(f"Added seed: (y={iy}, x={ix})")

    def update_image(self):
        self.image_view.setImage(self.grid.T, autoLevels=True)  # .T for correct orientation
        self.seed_points = []

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
