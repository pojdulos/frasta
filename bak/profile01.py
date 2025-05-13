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
adjusted_grid_smooth = gaussian_filter(adjusted_grid, sigma=sigma)
reference_grid_smooth = gaussian_filter(reference_grid, sigma=sigma)
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

        self.image_view = pg.ImageView()
        self.image_view2 = pg.ImageView()
        self.image_view3 = pg.ImageView()

        for view in [self.image_view, self.image_view2, self.image_view3]:
            view.ui.histogram.hide()
            view.ui.roiBtn.hide()
            view.ui.menuBtn.hide()
            view.getView().setBackgroundColor('w')

        left_layout = QtWidgets.QVBoxLayout()
        left_layout.addWidget(self.image_view)
        left_layout.addWidget(self.image_view2)
        left_layout.addWidget(self.image_view3)
        layout.addLayout(left_layout)

        right_layout = QtWidgets.QVBoxLayout()
        self.plot_widget = pg.PlotWidget()
        size_x_mm = reference_grid.shape[0] * metadata['pixel_size_x_mm']
        self.plot_widget.getPlotItem().getViewBox().setRange(xRange=(0, size_x_mm))
        self.plot_widget.getPlotItem().getViewBox().setMouseEnabled(x=True, y=False)
        right_layout.addWidget(self.plot_widget)

        self.slider_angle = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_angle.setRange(0, 180)
        self.slider_angle.valueChanged.connect(self.update_plot)
        right_layout.addWidget(self.slider_angle)

        self.slider_separation = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_separation.setRange(0, 300)
        self.slider_separation.valueChanged.connect(self.update_plot)
        right_layout.addWidget(self.slider_separation)

        layout.addLayout(right_layout)
        layout.setStretch(0, 1)
        layout.setStretch(1, 3)

        self.cursor_lines = []
        self.annotations = []
        self.update_plot()
        self.plot_widget.scene().sigMouseMoved.connect(self.on_mouse_move)

    def update_plot(self):
        angle = self.slider_angle.value()
        separation = self.slider_separation.value()
        angle_rad = np.deg2rad(angle)
        dx = int(length * np.cos(angle_rad))
        dy = int(length * np.sin(angle_rad))
        x1, y1 = center[1] - dx, center[0] - dy
        x2, y2 = center[1] + dx, center[0] + dy

        rr, cc = line(y1, x1, y2, x2)
        diffs = np.sqrt(np.diff(rr)**2 + np.diff(cc)**2)
        positions_line = np.concatenate(([0], np.cumsum(diffs))) * metadata['pixel_size_x_mm']

        self.reference_profile = reference_grid_smooth[rr, cc]
        self.adjusted_profile = (adjusted_grid_corrected + separation)[rr, cc]

        difference = reference_grid_smooth - (adjusted_grid_corrected + separation)
        binary_contact = (difference <= 0) & ~np.isnan(difference)
        binary_contact = np.where(ring_mask, binary_contact, True)

        self.image_view.setImage(binary_contact.T, autoLevels=True)
        self.image_view2.setImage(adjusted_grid_corrected, autoLevels=True)
        self.image_view3.setImage(reference_grid_smooth, autoLevels=True)


        # Usuń starą linię jeśli istnieje
        if hasattr(self, 'line_roi'):
            self.image_view3.getView().removeItem(self.line_roi)

        # Linia przekroju (od x1,y1 do x2,y2)
        self.line_roi = pg.LineROI([x1, y1], [x2, y2], pen=pg.mkPen('r', width=1), width=1)
        self.image_view3.getView().addItem(self.line_roi)
        self.line_roi.setZValue(10)

        self.plot_widget.clear()
        self.plot_widget.plot(positions_line, self.reference_profile, pen=pg.mkPen('g', width=2))
        self.plot_widget.plot(positions_line, self.adjusted_profile, pen=pg.mkPen('b', width=2))

    def on_mouse_move(self, pos):
        angle = self.slider_angle.value()
        separation = self.slider_separation.value()
        angle_rad = np.deg2rad(angle)
        dx = int(length * np.cos(angle_rad))
        dy = int(length * np.sin(angle_rad))
        x1, y1 = center[1] - dx, center[0] - dy
        x2, y2 = center[1] + dx, center[0] + dy
        rr, cc = line(y1, x1, y2, x2)
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
            height_diff = self.reference_profile[idx] - self.adjusted_profile[idx]
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
