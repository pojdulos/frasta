import sys
import os
import numpy as np
import h5py
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from scipy.ndimage import gaussian_filter
from skimage.draw import line
from sklearn.linear_model import LinearRegression

app = QtWidgets.QApplication([])

# Wczytanie danych z pliku HDF5
current_dir = os.path.dirname(os.path.realpath(__file__))
h5_path = os.path.join(current_dir, "source_data/aligned.h5")

with h5py.File(h5_path, "r") as f:
    reference_grid = f["scan1"][:]
    adjusted_grid = f["scan2"][:]

# Zakładamy cały obszar jako ważny
ring_mask = ~np.isnan(reference_grid)

# Parametry
metadata = {"pixel_size_x_mm": 0.00262}
sigma = 2.0
separation = 0
n_rows, n_cols = adjusted_grid.shape
center = (n_rows // 2, n_cols // 2)
length = min(n_rows, n_cols) // 2 - 10

def remove_relative_tilt(reference, target, mask):
    difference = reference - target
    rows, cols = difference.shape
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    XX = X[mask].flatten()
    YY = Y[mask].flatten()
    ZZ = difference[mask].flatten()
    valid_mask = ~np.isnan(ZZ)
    XX, YY, ZZ = XX[valid_mask], YY[valid_mask], ZZ[valid_mask]
    if len(ZZ) == 0:
        raise ValueError("Brak ważnych danych do regresji - wszystkie punkty zawierały NaN")
    features = np.vstack((XX, YY)).T
    model = LinearRegression().fit(features, ZZ)
    tilt_plane = model.predict(np.vstack((X.flatten(), Y.flatten())).T).reshape(difference.shape)
    return target + tilt_plane

# Przetwarzanie
adjusted_grid_smooth = adjusted_grid #gaussian_filter(adjusted_grid, sigma=sigma)
reference_grid_smooth = reference_grid #gaussian_filter(reference_grid, sigma=sigma)
offset_correction = np.nanmean(reference_grid_smooth - adjusted_grid_smooth)
adjusted_grid_corrected = adjusted_grid_smooth + offset_correction
adjusted_grid_corrected = remove_relative_tilt(reference_grid_smooth, adjusted_grid_corrected, ring_mask)

class ProfilViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interaktywna analiza przekrojów")
        self.setGeometry(100, 100, 1000, 600)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QHBoxLayout()
        central_widget.setLayout(layout)

        self.line_roi = None

        # Tworzenie widoków
        self.image_view = pg.ImageView()      # obraz 1 – binarny (teraz po prawej)
        self.image_view2 = pg.ImageView()     # obraz 2 – adjusted
        self.image_view3 = pg.ImageView()     # obraz 3 – reference z ROI

        for view in [self.image_view, self.image_view2, self.image_view3]:
            view.ui.histogram.hide()
            view.ui.roiBtn.hide()
            view.ui.menuBtn.hide()
            view.getView().setBackgroundColor('w')

        # Lewa kolumna – adjusted + reference
        left_layout = QtWidgets.QHBoxLayout()
        #self.image_view2.setMaximumWidth(200)
        #self.image_view3.setMaximumWidth(200)
        left_layout.addWidget(self.image_view2)
        left_layout.addWidget(self.image_view3)

        # Środkowa kolumna – wykres i suwaki
        center_layout = QtWidgets.QVBoxLayout()
        self.plot_widget = pg.PlotWidget()
        size_x_mm = reference_grid.shape[0] * metadata['pixel_size_x_mm']
        self.plot_widget.getPlotItem().getViewBox().setRange(xRange=(0, size_x_mm))
        self.plot_widget.getPlotItem().getViewBox().setMouseEnabled(x=True, y=False)
        center_layout.addWidget(self.plot_widget)

        self.slider_separation = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_separation.setRange(-500, 500)
        self.slider_separation.valueChanged.connect(self.update_plot)
        center_layout.addWidget(self.slider_separation)

        # Prawa kolumna – binary image
        right_layout = QtWidgets.QVBoxLayout()

        right_layout.addLayout(left_layout)
        #self.image_view.setMaximumWidth(400)
        right_layout.addWidget(self.image_view)

        # Dodanie do głównego layoutu
        layout.addLayout(center_layout)
        layout.addLayout(right_layout)

        layout.setStretch(0, 2)
        layout.setStretch(1, 3)
        layout.setStretch(2, 2)

        print(reference_grid.shape)
        print(adjusted_grid.shape)
        
        height, width = reference_grid_smooth.shape
        self.x1, self.y1 = 0, 0
        self.x2, self.y2 = width - 1, height - 1

        self.line_roi = pg.LineROI([self.x1, self.y1], [self.x2, self.y2], pen=pg.mkPen('r', width=2), width=1)
        self.line_roi.handles[2]['type'] = 'center'
        self.line_roi.sigRegionChanged.connect(self.update_profile_from_roi)
        self.image_view.getView().addItem(self.line_roi)
        self.line_roi.setZValue(10)

        self.cursor_lines = []
        self.annotations = []
        self.update_plot()
        self.plot_widget.scene().sigMouseMoved.connect(self.on_mouse_move)

    def update_plot(self):
        # Zaktualizuj widoki obrazów
        separation = self.slider_separation.value()
        difference = reference_grid_smooth - (adjusted_grid_corrected + separation)
        binary_contact = (difference <= 0) & ~np.isnan(difference)
        binary_contact = np.where(ring_mask, binary_contact, True)

        self.image_view.setImage(binary_contact.T, autoLevels=True)
        self.image_view2.setImage(adjusted_grid_corrected.T, autoLevels=True)
        self.image_view3.setImage(reference_grid_smooth.T, autoLevels=True)

        self.update_profile_from_roi()

    def update_profile_from_roi(self):
        handle0 = self.line_roi.getHandles()[0].pos()
        handle1 = self.line_roi.getHandles()[1].pos()

        pos1 = self.line_roi.mapToParent(handle0)
        pos2 = self.line_roi.mapToParent(handle1)

        self.y1, self.x1 = int(pos1.y()), int(pos1.x())
        self.y2, self.x2 = int(pos2.y()), int(pos2.x())

        rr, cc = line(self.y1, self.x1, self.y2, self.x2)

        # Ogranicz indeksy do rozmiaru obrazu
        rr = np.clip(rr, 0, reference_grid_smooth.shape[0] - 1)
        cc = np.clip(cc, 0, reference_grid_smooth.shape[1] - 1)

        # Pobierz profile
        profile_ref = reference_grid_smooth[rr, cc]
        profile_adj = (adjusted_grid_corrected + self.slider_separation.value())[rr, cc]

        # Stwórz linię pozycji w mm
        length_profile = len(profile_ref)
        positions_line = np.arange(length_profile) * metadata['pixel_size_x_mm']

        # Zapisz do pól
        self.reference_profile = profile_ref
        self.adjusted_profile = profile_adj

        # Odśwież wykres
        self.plot_widget.clear()
        self.plot_widget.plot(positions_line, self.reference_profile, pen=pg.mkPen('g', width=2))
        self.plot_widget.plot(positions_line, self.adjusted_profile, pen=pg.mkPen('b', width=2))

    def on_mouse_move(self, pos):
        rr, cc = line(self.y1, self.x1, self.y2, self.x2)

        # Ogranicz indeksy do rozmiaru obrazu
        # rr = np.clip(rr, 0, reference_grid_smooth.shape[0] - 1)
        # cc = np.clip(cc, 0, reference_grid_smooth.shape[1] - 1)

        diffs = np.sqrt(np.diff(rr)**2 + np.diff(cc)**2)
        positions_line = np.concatenate(([0], np.cumsum(diffs))) * metadata['pixel_size_x_mm']

        if self.plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            x_pos = mouse_point.x()
            for item in self.cursor_lines + self.annotations:
                self.plot_widget.removeItem(item)
            self.cursor_lines.clear()
            self.annotations.clear()

            idx = np.argmin(np.abs(positions_line - x_pos))
            height_diff = self.reference_profile[idx] - (self.adjusted_profile[idx])

            vline = pg.InfiniteLine(pos=x_pos, angle=90, pen=pg.mkPen('r', width=1, style=QtCore.Qt.DashLine))
            self.plot_widget.addItem(vline)
            self.cursor_lines.append(vline)
            text = pg.TextItem(f"{height_diff:.2f} μm", color='g', anchor=(0.5, 1))
            text.setPos(x_pos, self.reference_profile[idx])
            self.plot_widget.addItem(text)
            self.annotations.append(text)

if __name__ == '__main__':
    viewer = ProfilViewer()
    viewer.show()
    sys.exit(app.exec_())
