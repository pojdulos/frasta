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
from math import atan, degrees
import pyqtgraph.opengl as gl

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

def resource_path(relative_path):
    """Zwraca absolutną ścieżkę do zasobu (działa i w exe, i w .py)"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


class Profile3DWindow(QtWidgets.QWidget):
    def __init__(self, reference_grid, adjusted_grid, line_points=None, separation=0, auto_center_z=True, parent=None):
        super().__init__(parent)
        self.setWindowTitle("3D View of Grids and Profile Plane")
        self.resize(900, 700)
        layout = QtWidgets.QVBoxLayout(self)

        # ---- Panel z checkboxami ----
        ctrl_layout = QtWidgets.QHBoxLayout()
        self.checkbox_ref = QtWidgets.QCheckBox("Show Reference Surface")
        self.checkbox_ref.setChecked(True)
        self.checkbox_adj = QtWidgets.QCheckBox("Show Adjusted Surface")
        self.checkbox_adj.setChecked(True)
        self.checkbox_line = QtWidgets.QCheckBox("Show Profile Line")
        self.checkbox_line.setChecked(True)
        self.checkbox_plane = QtWidgets.QCheckBox("Show Section Plane")
        self.checkbox_plane.setChecked(True)
        ctrl_layout.addWidget(self.checkbox_ref)
        ctrl_layout.addWidget(self.checkbox_adj)
        ctrl_layout.addWidget(self.checkbox_line)
        ctrl_layout.addWidget(self.checkbox_plane)
        layout.addLayout(ctrl_layout)
        # ---- Koniec panelu ----

        self.view = gl.GLViewWidget()
        layout.addWidget(self.view)
        self.view.setCameraPosition(distance=200)

        # Przechowywane referencje do elementów
        self.surface_ref_item = None
        self.surface_adj_item = None
        self.ref_profile_line_item = None
        self.adj_profile_line_item = None
        self.cross_plane_item = None


        #step = max(1, min(reference_grid.shape[0], reference_grid.shape[1]) // 512)
        step = 5

        print(f"step={step}")        
        ys = np.arange(0, reference_grid.shape[0], step)
        xs = np.arange(0, reference_grid.shape[1], step)
        Z_ref = reference_grid[np.ix_(ys, xs)]
        Z_adj = adjusted_grid[np.ix_(ys, xs)] + separation

        print(f"Zref: {Z_ref.shape}, Zadj: {Z_adj.shape}")

        Z_MAX = 1e6
        Z_MIN = -1e6

        # Zamień niepoprawne na NaN
        Z_ref = np.where(np.isfinite(Z_ref), Z_ref, np.nan)
        Z_adj = np.where(np.isfinite(Z_adj), Z_adj, np.nan)

        # Zamień punkty spoza zakresu na NaN (nie będą rysowane)
        # Z_ref = np.where((Z_ref > Z_MAX) | (Z_ref < Z_MIN), np.nan, Z_ref)
        # Z_adj = np.where((Z_adj > Z_MAX) | (Z_adj < Z_MIN), np.nan, Z_adj)
        Z_ref = np.where((Z_ref > Z_MAX) | (Z_ref < Z_MIN), 0.0, Z_ref)
        Z_adj = np.where((Z_adj > Z_MAX) | (Z_adj < Z_MIN), 0.0, Z_adj)

        Z_ref = Z_ref / 1000.0  # teraz Z w mm, jeśli wcześniej było w μm
        Z_adj = Z_adj / 1000.0

        # Jeśli cała siatka NaN, NIE rysuj jej!
        if not np.all(np.isnan(Z_ref)):
            self.surface_ref_item = gl.GLSurfacePlotItem(
                x=xs, y=ys, z=Z_ref.T, color=(0,1,0,1), shader='shaded'
            )
            self.view.addItem(self.surface_ref_item)

        if not np.all(np.isnan(Z_adj)):
            self.surface_adj_item = gl.GLSurfacePlotItem(
                x=xs, y=ys, z=Z_adj.T, color=(0.2,0.3,1,1), shader='shaded'
            )
            self.view.addItem(self.surface_adj_item)

        z_min = min(np.nanmin(Z_ref), np.nanmin(Z_adj))
        z_max = max(np.nanmax(Z_ref), np.nanmax(Z_adj))
        self.cross_plane_item = self.add_cross_section_plane(line_points, z_min, z_max)

        # Dodaj linię profilu
        if line_points is not None and len(line_points) > 1:
            pts = np.array(line_points)
            ref_prof = reference_grid[pts[:,1], pts[:,0]]
            ref_prof = ref_prof / 1000.0

            self.ref_profile_line_item = gl.GLLinePlotItem(
                pos=np.column_stack((pts[:,0], pts[:,1], ref_prof)),
                color=(0,0.4,0,1), width=2
                #, glOptions='opaque'
                #, antialias=True
            )
            self.view.addItem(self.ref_profile_line_item)

            adj_prof = adjusted_grid[pts[:,1], pts[:,0]] + separation
            adj_prof = adj_prof / 1000.0

            self.adj_profile_line_item = gl.GLLinePlotItem(
                pos=np.column_stack((pts[:,0], pts[:,1], adj_prof)),
                color=(0,0,1,1), width=2
                #, glOptions='opaque'
                #, antialias=True
            )
            self.view.addItem(self.adj_profile_line_item)


        # Połącz checkboxy z metodami
        self.checkbox_ref.stateChanged.connect(self.toggle_surface_ref)
        self.checkbox_adj.stateChanged.connect(self.toggle_surface_adj)
        self.checkbox_line.stateChanged.connect(self.toggle_profile_line)
        self.checkbox_plane.stateChanged.connect(self.toggle_cross_plane)

    def add_cross_section_plane(self, line_points, z_min, z_max):
        """
        Rysuje półprzezroczystą płaszczyznę przekroju (jeden prostokąt!) w 3D, wyznaczoną przez końce linii profilu.
        Zwraca obiekt mesh dla kontroli widoczności.
        """
        if line_points is None or len(line_points) < 2:
            return None
        p0 = np.array(line_points[0])
        p1 = np.array(line_points[-1])
        # Cztery wierzchołki prostokąta
        pts = np.array([
            [p0[0], p0[1], z_min],
            [p1[0], p1[1], z_min],
            [p1[0], p1[1], z_max],
            [p0[0], p0[1], z_max],
        ])
        faces = np.array([
            [0,1,2],
            [0,2,3],
        ])
        color = np.array([0.5,0.5,0.7,0.7])
        mesh = gl.GLMeshItem(vertexes=pts, faces=faces, glOptions='translucent', faceColors=np.tile(color, (2,1)), smooth=False, drawEdges=False)
        self.view.addItem(mesh)
        return mesh

    # Metody do przełączania widoczności
    def toggle_surface_ref(self, state):
        if self.surface_ref_item:
            self.surface_ref_item.setVisible(bool(state))

    def toggle_surface_adj(self, state):
        if self.surface_adj_item:
            self.surface_adj_item.setVisible(bool(state))

    def toggle_profile_line(self, state):
        if self.ref_profile_line_item:
            self.ref_profile_line_item.setVisible(bool(state))
        if self.adj_profile_line_item:
            self.adj_profile_line_item.setVisible(bool(state))

    def toggle_cross_plane(self, state):
        if self.cross_plane_item:
            self.cross_plane_item.setVisible(bool(state))


from PyQt5.QtCore import QThread, pyqtSignal

class ProfilWorker(QThread):
    # Sygnały (możesz dodać więcej, np. postęp)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, filepath, sigma, metadata):
        super().__init__()
        self.filepath = filepath
        self.sigma = sigma
        self.metadata = metadata

    def run(self):
        try:
            import numpy as np
            import h5py
            from scipy.ndimage import gaussian_filter
            from sklearn.linear_model import LinearRegression

            with h5py.File(self.filepath, "r") as f:
                reference_grid = f["scan1"][:]
                adjusted_grid = f["scan2"][:]
            reference_grid_smooth = gaussian_filter(reference_grid, sigma=self.sigma)
            adjusted_grid_smooth = gaussian_filter(adjusted_grid, sigma=self.sigma)
            valid_mask = ~np.isnan(reference_grid_smooth) & ~np.isnan(adjusted_grid_smooth)
            offset_correction = np.nanmean(reference_grid_smooth - adjusted_grid_smooth)
            adjusted_grid_corrected = adjusted_grid_smooth + offset_correction
            # Gotowe, zwróć wszystko w dict
            result = {
                "reference_grid": reference_grid,
                "adjusted_grid": adjusted_grid,
                "reference_grid_smooth": reference_grid_smooth,
                "adjusted_grid_smooth": adjusted_grid_smooth,
                "valid_mask": valid_mask,
                "adjusted_grid_corrected": adjusted_grid_corrected,
            }
            self.finished.emit(result)
        except Exception as e:
            import traceback
            self.error.emit(str(e) + '\n' + traceback.format_exc())



class ProfilViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive cross-sectional analysis")
        self.setGeometry(100, 100, 1000, 600)

        # --- PARAMETRY, metadane i domyślna ścieżka ---
        self.metadata = {"pixel_size_x_mm": 0.00138}
        self.sigma = 5.0
        self.separation = 0

        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')
        self.open_action = QtWidgets.QAction('Open...', self)
        self.open_action.triggered.connect(self.load_new_data)
        file_menu.addAction(self.open_action)
        file_menu.addSeparator()
        self.exit_action = QtWidgets.QAction('Exit', self)
        self.exit_action.triggered.connect(self.close)
        file_menu.addAction(self.exit_action)

        view_menu = menubar.addMenu('View')
        self.open_3d_action = QtWidgets.QAction('Show 3D view', self)
        self.open_3d_action.triggered.connect(self.show_3d_view)
        view_menu.addAction(self.open_3d_action)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QHBoxLayout()
        central_widget.setLayout(layout)

        # Środkowa kolumna – wykres i suwaki
        center_layout = QtWidgets.QVBoxLayout()
        # self.plot_widget = pg.PlotWidget()
        # size_x_mm = self.reference_grid.shape[0] * self.metadata['pixel_size_x_mm']
        # self.plot_widget.getPlotItem().getViewBox().setRange(xRange=(0, size_x_mm))
        # self.plot_widget.getPlotItem().getViewBox().setMouseEnabled(x=True, y=True)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.getPlotItem().getViewBox().setRange(xRange=(0, 1))
        self.plot_widget.getPlotItem().getViewBox().setMouseEnabled(x=True, y=True)
        center_layout.addWidget(self.plot_widget)

        # Prawa kolumna – binary image
        right_layout = QtWidgets.QVBoxLayout()
        self.image_view = create_image_view()
        self.image_view.setMinimumWidth(400)
        right_layout.addWidget(self.image_view)
        vb = self.image_view.getView()
        vb.setRange(
            xRange=(0, 1000),
            # yRange=(0, self.reference_grid.shape[0]-1),
            padding=0
        )
        self.image_view.getView().sigRangeChanged.connect(self.on_range_changed)

        sep_layout = QtWidgets.QHBoxLayout()
        self.spinbox_separation = QtWidgets.QSpinBox()
        self.spinbox_separation.setRange(-1000, 1000)
        self.spinbox_separation.setValue(self.separation)
        self.spinbox_separation.valueChanged.connect(self.update_plot)
        sep_layout.addWidget(QtWidgets.QLabel("Separation:"))
        sep_layout.addWidget(self.spinbox_separation)
        right_layout.addLayout(sep_layout)

        self.spinbox_window_mm = QtWidgets.QDoubleSpinBox()
        self.spinbox_window_mm.setRange(0.001, 5.0)
        self.spinbox_window_mm.setValue(0.5)
        self.spinbox_window_mm.setSingleStep(0.001)
        self.spinbox_window_mm.setDecimals(3)
        self.spinbox_window_mm.valueChanged.connect(self.update_plot)

        self.checkbox_snap = QtWidgets.QCheckBox("Snap to plot")
        self.checkbox_snap.setChecked(True)
        
        win_layout = QtWidgets.QHBoxLayout()
        win_layout.addWidget(QtWidgets.QLabel("Window size [mm]:"))
        win_layout.addWidget(self.spinbox_window_mm)
        win_layout.addWidget(self.checkbox_snap)

        right_layout.addLayout(win_layout)

        self.checkbox_tilt = QtWidgets.QCheckBox("Tilt correction")
        self.checkbox_tilt.setChecked(True)
        self.checkbox_tilt.stateChanged.connect(self.toggle_tilt)

        right_layout.addWidget(self.checkbox_tilt)

        # Dodanie do głównego layoutu
        layout.addLayout(center_layout)
        layout.addLayout(right_layout)

        # Pozostałe pola/zmienne
        # height, width = self.reference_grid_smooth.shape
        # self.x1, self.y1 = 0, 0
        # self.x2, self.y2 = width - 1, height - 1

        self.cursor_lines = []
        self.annotations = []
        self.mytest = []

        self.image_marker = None
        self.saved_points = []
        self.saved_point_markers = []

        self.plot_widget.scene().sigMouseMoved.connect(self.on_mouse_move)
        self.plot_widget.scene().sigMouseClicked.connect(self.on_plot_click)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)

        # self.statusBar().showMessage("Gotowy")

        # W __init__, po ustawieniu status/progress_bar, itp:
        self.statusBar().showMessage("Wczytywanie danych...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        QtWidgets.QApplication.processEvents()
        self.worker = ProfilWorker(resource_path("source_data/moj_test_aligned.h5"), self.sigma, self.metadata)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.error.connect(self.on_worker_error)

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self.centralWidget().setEnabled(False)
        self.open_action.setEnabled(False)
        self.worker.start()

    def show_3d_view(self):
        # Wyznacz linię profilu
        if hasattr(self, 'rr') and hasattr(self, 'cc'):
            line_points = list(zip(self.cc, self.rr))  # (x, y) dla linii
        else:
            line_points = None
        # Utwórz i pokaż okno 3D
        win = Profile3DWindow(self.reference_grid_smooth, self.adjusted_grid_corrected, line_points=line_points, separation=self.separation)
        win.show()
        # Zatrzymaj referencję, żeby okno nie znikało
        self._win3d = win


    def on_range_changed(self, viewbox, ranges):
        x_range, y_range = viewbox.viewRange()
        print(f"Ranges: {ranges}")
        print(f"Nowy zakres X: {x_range}, Y: {y_range}")

    def toggle_tilt(self, state):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self.centralWidget().setEnabled(False)
        self.open_action.setEnabled(False)
        offset_correction = np.nanmean(self.reference_grid_smooth - self.adjusted_grid_smooth)
        self.adjusted_grid_corrected = self.adjusted_grid_smooth + offset_correction
        if self.checkbox_tilt.isChecked():
            self.adjusted_grid_corrected = remove_relative_tilt(self.reference_grid_smooth, self.adjusted_grid_corrected, self.valid_mask)

        self.redraw_roi()
        self.update_plot()

        self.centralWidget().setEnabled(True)
        self.open_action.setEnabled(True)
        QtWidgets.QApplication.restoreOverrideCursor()

        
    def load_new_data(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Wybierz plik HDF5", "", "HDF5 files (*.h5);;Wszystkie pliki (*)")
        if fname:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            self.centralWidget().setEnabled(False)
            self.open_action.setEnabled(False)
            self.statusBar().showMessage("Wczytywanie danych...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            self.worker = ProfilWorker(fname, self.sigma, self.metadata)
            self.worker.finished.connect(self.on_worker_finished)
            self.worker.error.connect(self.on_worker_error)
            self.worker.start()

    def on_worker_finished(self, result):
        # Zaktualizuj wszystkie dane na podstawie wyniku z wątku
        self.centralWidget().setEnabled(True)
        self.open_action.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("Gotowy")

        self.reference_grid = result["reference_grid"]
        self.adjusted_grid = result["adjusted_grid"]
        self.reference_grid_smooth = result["reference_grid_smooth"]
        self.adjusted_grid_smooth = result["adjusted_grid_smooth"]
        self.valid_mask = result["valid_mask"]
        self.adjusted_grid_corrected = result["adjusted_grid_corrected"]
        if self.checkbox_tilt.isChecked():
            self.adjusted_grid_corrected = remove_relative_tilt(self.reference_grid_smooth, self.adjusted_grid_corrected, self.valid_mask)

        size_x_mm = self.reference_grid.shape[0] * self.metadata['pixel_size_x_mm']
        self.plot_widget.getPlotItem().getViewBox().setRange(xRange=(0, size_x_mm))

        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("Gotowy")
        # Reset ROI i odśwież GUI
        height, width = self.reference_grid_smooth.shape
        self.x1, self.y1 = 0, 0
        self.x2, self.y2 = width - 1, height - 1

        # Tutaj już możesz bezpiecznie!
        height, width = self.reference_grid_smooth.shape
        self.x1, self.y1 = 0, 0
        self.x2, self.y2 = width - 1, height - 1  

        self.redraw_roi()
        
        shape = self.update_plot()

        self.resize_image_view(shape)

        # aspect = shape[0]/shape[1]
        # div = (shape[1] // 500) if (aspect <= 1.0) else (aspect*shape[1] // 500)
        # self.image_view.setFixedSize(shape[1]//div,shape[0]//div)
        # self.image_view.update()
        # self.updateGeometry()

        vb = self.image_view.getView()
        vb.setAspectLocked(True)

        vb.setLimits( 
            yMin=0, yMax=shape[0]-1,
            xMin=0, xMax=shape[1]-1 
        )
        
        vb.setRange(
            xRange=(0, shape[1]-1),
            yRange=(0, shape[0]-1),
            padding=0
        )

        QtWidgets.QApplication.restoreOverrideCursor()

    def resize_image_view(self, shape):
        # shape = (height, width)
        height, width = shape
        aspect = width / height
        # bazowy wymiar, np. 700
        base = 500
        if aspect >= 1.0:
            w = base
            h = int(base / aspect)
        else:
            h = base
            w = int(base * aspect)
        self.image_view.setFixedSize(w, h)
        self.image_view.update()
        self.updateGeometry()

    def on_worker_error(self, msg):
        self.progress_bar.setVisible(False)
        QtWidgets.QApplication.restoreOverrideCursor()
        self.statusBar().showMessage("Błąd podczas przetwarzania!")
        QtWidgets.QMessageBox.critical(self, "Błąd", "Błąd podczas przetwarzania danych:\n" + msg)

    def update_plot(self):
        # Zaktualizuj widoki obrazów
        self.separation = self.spinbox_separation.value()
        valid_mask = ~np.isnan(self.reference_grid_smooth) & ~np.isnan(self.adjusted_grid_corrected)
        difference = self.reference_grid_smooth - (self.adjusted_grid_corrected + self.separation)
        binary_contact = (difference <= 0) & valid_mask

        self.image_view.setImage(binary_contact.T, autoRange=False, autoLevels=True)

        self.update_profile_from_roi()

        return binary_contact.shape

    def redraw_roi(self):
        if hasattr(self, 'line_roi'):
            self.image_view.getView().removeItem(self.line_roi)
        self.line_roi = pg.LineROI([self.x1, self.y1], [self.x2, self.y2], pen=pg.mkPen('r', width=2), width=1)
        self.line_roi.handles[2]['type'] = 'center'
        self.line_roi.sigRegionChanged.connect(self.update_profile_from_roi)
        self.image_view.getView().addItem(self.line_roi)
        self.line_roi.setZValue(10)

    def clamp_roi_to_image(self):
        img_shape = self.reference_grid_smooth.shape  # (rows, cols)
        h1 = self.line_roi.getHandles()[0].pos()
        h2 = self.line_roi.getHandles()[1].pos()
        pos1 = self.line_roi.mapToParent(h1).toPoint()
        pos2 = self.line_roi.mapToParent(h2).toPoint()
        self.x1 = min(max(pos1.x(), 0), img_shape[1] - 1)
        self.y1 = min(max(pos1.y(), 0), img_shape[0] - 1)
        self.x2 = min(max(pos2.x(), 0), img_shape[1] - 1)
        self.y2 = min(max(pos2.y(), 0), img_shape[0] - 1)
        if (pos1.x(), pos1.y(), pos2.x(), pos2.y()) != (self.x1, self.y1, self.x2, self.y2):
            self.redraw_roi()

    def update_profile_from_roi(self):
        self.clamp_roi_to_image()
        rr, cc = line(self.y1, self.x1, self.y2, self.x2)
        rr = np.clip(rr, 0, self.reference_grid_smooth.shape[0] - 1)
        cc = np.clip(cc, 0, self.reference_grid_smooth.shape[1] - 1)
        profile_ref = self.reference_grid_smooth[rr, cc]
        profile_adj = (self.adjusted_grid_corrected + self.separation)[rr, cc]
        valid_profile_mask = ~np.isnan(profile_ref) & ~np.isnan(profile_adj)
        self.rr = rr[valid_profile_mask]
        self.cc = cc[valid_profile_mask]
        profile_ref = profile_ref[valid_profile_mask]
        profile_adj = profile_adj[valid_profile_mask]
        positions_line = np.arange(len(rr))[valid_profile_mask] * self.metadata['pixel_size_x_mm']
        self.positions_line = positions_line
        self.reference_profile = profile_ref
        self.adjusted_profile = profile_adj
        self.plot_widget.clear()
        self.plot_widget.plot(positions_line, self.reference_profile, pen=pg.mkPen('g', width=2))
        self.plot_widget.plot(positions_line, self.adjusted_profile, pen=pg.mkPen('b', width=2))

    def print_saved_points(self):
        for i, pt in enumerate(self.saved_points):
            print(f"{i+1}: {pt}")

    def on_plot_click(self, event):
        if event.modifiers() == QtCore.Qt.ControlModifier:
            pos = event.scenePos()
            if self.plot_widget.sceneBoundingRect().contains(pos):
                mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
                x_pos = mouse_point.x()
                if self.positions_line[0] <= x_pos <= self.positions_line[-1]:
                    idx = np.argmin(np.abs(self.positions_line - x_pos))
                    if hasattr(self, 'rr') and hasattr(self, 'cc'):
                        y_img = self.rr[idx]
                        x_img = self.cc[idx]
                        ref_val = self.reference_profile[idx]
                        adj_val = self.adjusted_profile[idx]
                        pos_mm = self.positions_line[idx]
                        self.saved_points.append({
                            'profile_idx': idx,
                            'x_img': int(x_img),
                            'y_img': int(y_img),
                            'x_pos_mm': float(pos_mm),
                            'ref_val': float(ref_val),
                            'adj_val': float(adj_val),
                        })
                        marker = pg.ScatterPlotItem([x_img], [y_img], size=12, pen=pg.mkPen('g', width=2), brush=pg.mkBrush(0, 255, 255, 120), symbol='+')
                        self.image_view.getView().addItem(marker)
                        self.saved_point_markers.append(marker)
                        print("Saved point:", self.saved_points[-1])

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
                if hasattr(self, 'rr') and hasattr(self, 'cc'):
                    y_img = self.rr[idx]
                    x_img = self.cc[idx]
                    view = self.image_view.getView()
                    if self.image_marker is not None:
                        view.removeItem(self.image_marker)
                    self.image_marker = pg.ScatterPlotItem([x_img], [y_img], size=14, pen=pg.mkPen('m', width=2), brush=pg.mkBrush(255, 0, 255, 100))
                    view.addItem(self.image_marker)
                height_diff = self.reference_profile[idx] - self.adjusted_profile[idx]
                window_mm = self.spinbox_window_mm.value()
                pixel_size = self.metadata['pixel_size_x_mm']
                window_size = max(1, int(round(window_mm / pixel_size)))
                start = max(0, idx - window_size)
                end = min(len(self.positions_line), idx + window_size + 1)
                # Nachylenie profilu referencyjnego
                x_fit_ref = self.positions_line[start:end].reshape(-1, 1)
                y_fit_ref = self.reference_profile[start:end].reshape(-1, 1) / 1000.0
                reg_ref = LinearRegression().fit(x_fit_ref, y_fit_ref)
                slope_ref = reg_ref.coef_[0][0]
                angle_ref = degrees(atan(slope_ref))
                # Nachylenie profilu dopasowanego
                y_fit_adj = self.adjusted_profile[start:end].reshape(-1, 1) / 1000.0
                reg_adj = LinearRegression().fit(x_fit_ref, y_fit_adj)
                slope_adj = reg_adj.coef_[0][0]
                angle_adj = degrees(atan(slope_adj))
                delta_angle = angle_ref - angle_adj
                text1 = pg.TextItem(f"DIFF: {height_diff:.2f} μm", color='r', anchor=(0, 1))
                vb = self.plot_widget.getPlotItem().vb
                x_min, x_max = vb.viewRange()[0]
                y_min, y_max = vb.viewRange()[1]
                text1.setPos(x_min + 0.02 * (x_max - x_min), y_max - 0.05 * (y_max - y_min))
                self.plot_widget.addItem(text1)
                self.annotations.append(text1)
                text2 = pg.TextItem(f"ANGLE\nref: {angle_ref:.1f}°\nadj: {angle_adj:.1f}°\n  Δ: {delta_angle:.1f}°", color='y', anchor=(0, 1))
                text2.setPos(x_min + 0.02 * (x_max - x_min), y_max - 0.2 * (y_max - y_min))
                self.plot_widget.addItem(text2)
                self.annotations.append(text2)
                line_half_width_mm = window_mm / 2.0
                x0 = x_pos - line_half_width_mm
                x1 = x_pos + line_half_width_mm
                
                a = slope_ref
                if self.checkbox_snap.isChecked():
                    y_at_cursor = self.reference_profile[idx] / 1000.0
                    b = y_at_cursor - a * x_pos
                else:
                    b = reg_ref.intercept_[0]
                y0 = a * x0 + b
                y1 = a * x1 + b
                for item in self.mytest:
                    vb.removeItem(item)
                self.mytest.clear()
                line_ref = pg.PlotDataItem([x0, x1], [y0 * 1000, y1 * 1000], pen=pg.mkPen('y', width=2))
                vb.addItem(line_ref, ignoreBounds=True)
                self.annotations.append(line_ref)
                self.mytest.append(line_ref)
                
                a = slope_adj
                if self.checkbox_snap.isChecked():
                    y_at_cursor_adj = self.adjusted_profile[idx] / 1000.0
                    b_adj = y_at_cursor_adj - a * x_pos
                else:
                    b_adj = reg_adj.intercept_[0]
                y0 = a * x0 + b_adj
                y1 = a * x1 + b_adj
                line_adj = pg.PlotDataItem([x0, x1], [y0 * 1000, y1 * 1000], pen=pg.mkPen('y', width=2))
                vb.addItem(line_adj, ignoreBounds=True)
                self.annotations.append(line_adj)
                self.mytest.append(line_adj)
            else:
                vb = self.plot_widget.getPlotItem().vb
                for item in self.mytest:
                    vb.removeItem(item)
                self.mytest.clear()
                view = self.image_view.getView()
                if self.image_marker is not None:
                    view.removeItem(self.image_marker)
                    self.image_marker = None

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    viewer = ProfilViewer()
    viewer.show()
    sys.exit(app.exec_())
