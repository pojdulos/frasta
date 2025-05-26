import sys
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from skimage.segmentation import flood
from scipy.interpolate import griddata

from PyQt5 import QtCore

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
            self.progress.emit(int(20 + 30 * (i*chunk_size/total))) # np. do 50% progresu na czytaniu
        df = pd.concat(chunks, ignore_index=True)
        x, y, z = df['x'].values, df['y'].values, df['z'].values

        # Automatyczne określenie kroku
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
            # Emit progress co 1% lub co 10000 punktów
            if idx % max(1, N//50) == 0:
                # 50-99% na gridowaniu
                self.progress.emit(50 + int(49 * idx / N))

        mask_dup = (counts > 1)
        grid[mask_dup] = grid[mask_dup] / counts[mask_dup]
        xi_grid = np.linspace(x_min, x_max, grid_size_x)
        yi_grid = np.linspace(y_min, y_max, grid_size_y)
        self.progress.emit(100)
        self.finished.emit(grid, xi_grid, yi_grid, px_x, px_y, x, y, z)


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

    def open_file(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open file", "", "CSV, NPY, NPZ (*.csv *.npy *.npz)")
        if not fname:
            return
        if fname.endswith('.csv'):
            # Tworzymy modalny progress
            dlg = QtWidgets.QProgressDialog("Wczytywanie i gridowanie...", None, 0, 100, self)
            dlg.setWindowModality(QtCore.Qt.ApplicationModal)
            dlg.setAutoClose(True)
            dlg.setCancelButton(None)
            dlg.setValue(0)
            self.worker = GridWorker(fname)
            self.thread = QtCore.QThread()
            self.worker.moveToThread(self.thread)
            self.worker.progress.connect(dlg.setValue)
            self.worker.finished.connect(self.grid_ready)
            self.worker.finished.connect(self.thread.quit)
            self.thread.started.connect(self.worker.process)
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
                self.orig_data = (data['x_data'], data['y_data'], data['z_data'])
            else:
                QtWidgets.QMessageBox.warning(self, "Format error", "NPZ does not contain a grid.")
                return
        else:
            QtWidgets.QMessageBox.warning(self, "Unknown format", "Unsupported file type.")
            return
        self.update_image()

    def grid_ready(self, grid, xi, yi, px_x, px_y, x, y, z):
        self.grid = grid
        self.xi = xi
        self.yi = yi
        self.px_x = px_x
        self.px_y = px_y
        self.orig_data = (x, y, z)
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
        
        print(f"seed_points: {self.seed_points}")
        #return
        # a,b = self.grid.shape
        # self.seed_points = [(50,50),(a-50,50),(50,b-50),(a-50,b-50)]

        # print(f"seed_points: {self.seed_points}")

        tst = np.isnan(self.grid)
        print(f"tst shape: {tst.shape}")
        print("Liczba nan: ", np.count_nonzero(tst))
        for (iy, ix) in self.seed_points:
            filled = flood(tst, seed_point=(iy, ix))
            tst[filled] = False
            print("Liczba nan: ", np.count_nonzero(tst))

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
        vb = self.image_view.getView()
        mouse_point = vb.mapSceneToView(event.scenePos())
        x = int(round(mouse_point.x()))
        y = int(round(mouse_point.y()))
        # Dla grid.T – sprawdź czy nie zamienić x <-> y!
        if 0 <= x < self.grid.shape[1] and 0 <= y < self.grid.shape[0]:
            self.seed_points.append((y, x))  # (wiersz, kolumna) zgodnie z grid.shape
            scatter = pg.ScatterPlotItem([x], [y], size=10, brush=pg.mkBrush('r'))
            vb.addItem(scatter)
            print(f"Klik: x={x}, y={y}")
        else:
            print("Klik poza gridem!")

    def mouse_clicked1(self, event):
        if self.grid is None:
            return
        # Przekształć pozycję myszy z ekranu/sceny na współrzędne fizyczne (świata)
        vb = self.image_view.getView()
        mouse_point = vb.mapSceneToView(event.scenePos())
        x, y = mouse_point.x(), mouse_point.y()

        print("Grid shape:", self.grid.shape)
        print("xi: od", self.xi[0], "do", self.xi[-1], "len:", len(self.xi))
        print("yi: od", self.yi[0], "do", self.yi[-1], "len:", len(self.yi))

        # Przelicz na indeksy gridu:
        # self.xi - fizyczne X (np. mm), długość = liczba kolumn
        # self.yi - fizyczne Y, długość = liczba wierszy
        
        # ix = np.abs(self.xi - x).argmin()
        # iy = np.abs(self.yi - y).argmin()


        ix = np.abs(self.yi - x).argmin()   # 0...6086
        iy = np.abs(self.xi - y).argmin()   # 0...7873

        disp_y = self.xi[iy]   # self.xi ma 7874 elementy, iy mieści się w tym zakresie
        disp_x = self.yi[ix]   # self.yi ma 6087 elementów, ix mieści się w tym zakresie

        self.seed_points.append((ix, iy))   # indeksy do flooda

        scatter = pg.ScatterPlotItem([disp_x], [disp_y], size=10, brush=pg.mkBrush('r'))
        self.image_view.addItem(scatter)
        print(f"Kliknięto X={x:.2f} Y={y:.2f} -> indeks (iy={iy}, ix={ix})")

    def update_image(self):
        self.image_view.setImage(self.grid.T, autoLevels=True)  # .T for correct orientation
        self.seed_points = []

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
