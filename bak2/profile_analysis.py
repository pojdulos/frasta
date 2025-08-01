import sys
import os
import numpy as np
import h5py
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from scipy.ndimage import gaussian_filter
from skimage.draw import line
from sklearn.linear_model import LinearRegression
from PyQt5.QtCore import QPointF

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

def create_image_view():
    view = pg.ImageView()
    view.ui.histogram.hide()
    view.ui.roiBtn.hide()
    view.ui.menuBtn.hide()
    view.getView().setBackgroundColor('w')
    return view

class ProfilViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive cross-sectional analysis")
        self.setGeometry(100, 100, 1000, 600)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QHBoxLayout()
        central_widget.setLayout(layout)


        # Środkowa kolumna – wykres i suwaki
        center_layout = QtWidgets.QVBoxLayout()
        self.plot_widget = pg.PlotWidget()
        size_x_mm = reference_grid.shape[0] * metadata['pixel_size_x_mm']
        self.plot_widget.getPlotItem().getViewBox().setRange(xRange=(0, size_x_mm))
        # self.plot_widget.getPlotItem().getViewBox().setMouseEnabled(x=True, y=False)
        self.plot_widget.getPlotItem().getViewBox().setMouseEnabled(x=True, y=True)
        center_layout.addWidget(self.plot_widget)

        # Prawa kolumna – binary image
        right_layout = QtWidgets.QVBoxLayout()
        
        self.image_view = create_image_view()
        self.image_view.setMinimumWidth(400)
        right_layout.addWidget(self.image_view)

        
        sep_layout = QtWidgets.QHBoxLayout()
        self.spinbox_separation = QtWidgets.QSpinBox()
        self.spinbox_separation.setRange(-1000, 1000)
        self.spinbox_separation.valueChanged.connect(self.update_plot)
        
        sep_layout.addWidget(QtWidgets.QLabel("Separation:"))
        sep_layout.addWidget(self.spinbox_separation)

        right_layout.addLayout(sep_layout)

        # Dodanie do głównego layoutu
        layout.addLayout(center_layout)
        layout.addLayout(right_layout)

        height, width = reference_grid_smooth.shape
        self.x1, self.y1 = 0, 0
        self.x2, self.y2 = width - 1, height - 1

        self.cursor_lines = []
        self.annotations = []

        self.redraw_roi()
        self.update_plot()

        self.plot_widget.scene().sigMouseMoved.connect(self.on_mouse_move)

    def update_plot(self):
        # Zaktualizuj widoki obrazów
        separation = self.spinbox_separation.value()

        valid_mask = ~np.isnan(reference_grid_smooth) & ~np.isnan(adjusted_grid_corrected)
        difference = reference_grid_smooth - (adjusted_grid_corrected + separation)
        binary_contact = (difference <= 0) & valid_mask

        self.image_view.setImage(binary_contact.T, autoLevels=True)
        
        self.update_profile_from_roi()

    def redraw_roi(self):
        if hasattr(self, 'line_roi'):
            self.image_view.getView().removeItem(self.line_roi)

        self.line_roi = pg.LineROI([self.x1, self.y1], [self.x2, self.y2], pen=pg.mkPen('r', width=2), width=1)
        self.line_roi.handles[2]['type'] = 'center'
        self.line_roi.sigRegionChanged.connect(self.update_profile_from_roi)
        self.image_view.getView().addItem(self.line_roi)
        self.line_roi.setZValue(10)

    def clamp_roi_to_image(self):
        img_shape = reference_grid_smooth.shape  # (rows, cols)

        h1 = self.line_roi.getHandles()[0].pos()
        h2 = self.line_roi.getHandles()[1].pos()

        pos1 = self.line_roi.mapToParent(h1).toPoint()
        pos2 = self.line_roi.mapToParent(h2).toPoint()
        
        self.x1 = min(max(pos1.x(), 0),img_shape[1]-1)
        self.y1 = min(max(pos1.y(), 0),img_shape[0]-1)

        self.x2 = min(max(pos2.x(), 0),img_shape[1]-1)
        self.y2 = min(max(pos2.y(), 0),img_shape[0]-1)

        if (pos1.x(), pos1.y(), pos2.x(), pos2.y()) != (self.x1, self.y1, self.x2, self.y2):
            self.redraw_roi()


    def update_profile_from_roi(self):
        self.clamp_roi_to_image()

        rr, cc = line(self.y1, self.x1, self.y2, self.x2)

        # Ogranicz indeksy do rozmiaru obrazu
        rr = np.clip(rr, 0, reference_grid_smooth.shape[0] - 1)
        cc = np.clip(cc, 0, reference_grid_smooth.shape[1] - 1)

        # Pobierz profile
        profile_ref = reference_grid_smooth[rr, cc]
        profile_adj = (adjusted_grid_corrected + self.spinbox_separation.value())[rr, cc]

        # Maska: tylko tam, gdzie obie wartości są dostępne
        valid_profile_mask = ~np.isnan(profile_ref) & ~np.isnan(profile_adj)

        # Filtruj profile i pozycje
        profile_ref = profile_ref[valid_profile_mask]
        profile_adj = profile_adj[valid_profile_mask]
        positions_line = np.arange(len(rr))[valid_profile_mask] * metadata['pixel_size_x_mm']

        self.positions_line = positions_line

        # Zapisz do pól
        self.reference_profile = profile_ref
        self.adjusted_profile = profile_adj

        # Odśwież wykres
        self.plot_widget.clear()
        self.plot_widget.plot(positions_line, self.reference_profile, pen=pg.mkPen('g', width=2))
        self.plot_widget.plot(positions_line, self.adjusted_profile, pen=pg.mkPen('b', width=2))

    def on_mouse_move(self, pos):
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            x_pos = mouse_point.x()
            for item in self.cursor_lines + self.annotations:
                self.plot_widget.removeItem(item)
            self.cursor_lines.clear()
            self.annotations.clear()

            vline = pg.InfiniteLine(pos=x_pos, angle=90, pen=pg.mkPen('r', width=1, style=QtCore.Qt.DashLine))
            self.plot_widget.addItem(vline)
            self.cursor_lines.append(vline)

            positions_line = self.positions_line
            if positions_line[0] <= x_pos <= positions_line[-1]:
                idx = np.argmin(np.abs(positions_line - x_pos))
                height_diff = self.reference_profile[idx] - self.adjusted_profile[idx]
                text = pg.TextItem(f"{height_diff:.2f} μm", color='g', anchor=(0.5, 1))
                text.setPos(x_pos, self.reference_profile[idx])
                self.plot_widget.addItem(text)
                self.annotations.append(text)



app = QtWidgets.QApplication([])

# Wczytanie danych z pliku HDF5
current_dir = os.path.dirname(os.path.realpath(__file__))
h5_path = os.path.join(current_dir, "source_data/aligned.h5")

print(current_dir, h5_path)

with h5py.File(h5_path, "r") as f:
    reference_grid = f["scan1"][:]
    adjusted_grid = f["scan2"][:]

# Parametry
metadata = {"pixel_size_x_mm": 0.00138}
sigma = 5.0
separation = 0

# Przetwarzanie
#adjusted_grid_smooth = adjusted_grid
adjusted_grid_smooth = gaussian_filter(adjusted_grid, sigma=sigma)
#reference_grid_smooth = reference_grid
reference_grid_smooth = gaussian_filter(reference_grid, sigma=sigma)

valid_mask = ~np.isnan(reference_grid_smooth) & ~np.isnan(adjusted_grid_smooth)
        
offset_correction = np.nanmean(reference_grid_smooth - adjusted_grid_smooth)
adjusted_grid_corrected = adjusted_grid_smooth + offset_correction
adjusted_grid_corrected = remove_relative_tilt(reference_grid_smooth, adjusted_grid_corrected, valid_mask)


if __name__ == '__main__':
    viewer = ProfilViewer()
    viewer.show()
    sys.exit(app.exec_())
