import sys
import h5py
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QIcon
from functools import partial

from profileViewer import ProfileViewer
from overlayViewer import OverlayViewer
from aboutDialog import AboutDialog
from scanTab import ScanTab
from gridData import GridData

class GridWorker(QtCore.QObject):
    progress = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal(object, object, object, float, float) # grid, xi, yi, px_x, px_y

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

        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        grid_size_x = int((x_max - x_min) / px_x) + 1
        grid_size_y = int((y_max - y_min) / px_y) + 1

        grid = np.full((grid_size_y, grid_size_x), np.nan, dtype=np.float64)
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
        self.finished.emit(grid, xi_grid, yi_grid, px_x, px_y) #, x, y, z)

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
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self.close_tab)
        self.setCentralWidget(self.tabs)

        self.create_actions()
        self.connect_actions()
        self.create_menubar()
        self.create_toolbar()

        self.shared_roi = None  # będzie przechowywać jedną instancję CircleROI

        self.worker = None
        self.thread = None

        self.tabs.currentChanged.connect(self.move_roi_to_current_tab)

    def apply_roi_mask(self, inside):
        tab = self.current_tab()
        if tab is None or tab.grid is None or self.shared_roi is None or not self.shared_roi.isVisible():
            return
        pos = self.shared_roi.pos()
        size = self.shared_roi.size()
        cx = pos.x() + size[0]/2
        cy = pos.y() + size[1]/2
        r = size[0]/2
        h, w = tab.grid.shape
        Y, X = np.ogrid[:h, :w]
        mask = ((X - cx) ** 2 + (Y - cy) ** 2) <= r**2
        if inside:
            tab.set_mask(~mask)
        else:
            tab.set_mask(mask)

    def del_inside_mask(self):
        self.apply_roi_mask(True)

    def del_outside_mask(self):
        self.apply_roi_mask(False)

    def move_roi_to_current_tab(self, idx):
        if self.shared_roi is None or not self.shared_roi.isVisible():
            return
        # Usuń ROI z poprzedniego image_view
        for i in range(self.tabs.count()):
            tab = self.tabs.widget(i)
            tab.image_view.getView().removeItem(self.shared_roi)
        # Dodaj do bieżącego
        tab = self.tabs.widget(idx)
        tab.image_view.getView().addItem(self.shared_roi)
        self.shared_roi.show()

    def show_circle_roi(self):
        tab = self.current_tab()
        if tab is None or tab.grid is None:
            return
        
        if not self.shared_roi is None and self.shared_roi.isVisible():
            self.shared_roi.setVisible(False)
            return
        
        if self.shared_roi is None:
            import pyqtgraph as pg
            h, w = tab.grid.shape
            self.shared_roi = pg.CircleROI([w//2-50, h//2-50], [100, 100], pen=pg.mkPen('g', width=2))
            self.shared_roi.setZValue(100)
            # Dodaj slot do wykrywania zmian, jeśli chcesz
        tab.image_view.getView().addItem(self.shared_roi)
        self.shared_roi.show()

    def close_tab(self, index):
        widget = self.tabs.widget(index)
        if widget is not None:
            self.tabs.removeTab(index)
            widget.deleteLater()

    def create_actions(self):
        self.actions = { 
            "open": QtWidgets.QAction("Open...", self),
            "save_scan": QtWidgets.QAction("Save current scan...", self),
            "save_multi": QtWidgets.QAction("Save multiple scans...", self),
            "fill": QtWidgets.QAction("Fill holes", self),
            "flip": QtWidgets.QAction("Flip & reverse", self),
            "zero": QtWidgets.QAction("Set zero point", self),
            "tilt": QtWidgets.QAction("Set tilt", self),
            "colormap": QtWidgets.QAction("Toggle colormap", self),
            "view3d":  QtWidgets.QAction("View 3d...", self),
            "compare": QtWidgets.QAction("Scan positioning...", self),
            "profile": QtWidgets.QAction("Profile analysis...", self),
            "about": QtWidgets.QAction("About...", self),
            "exit": QtWidgets.QAction("Exit", self),
        }

        # create_actions
        self.actions["del_outside"] = QtWidgets.QAction("outside of the mask", self)
        self.actions["del_inside"] = QtWidgets.QAction("inside of the mask", self)
        self.actions["show_mask"] = QtWidgets.QAction("Show/hide the mask", self)



        self.actions["open"].setIcon(QIcon("icons/icons8-open-file1-50.png"))
        self.actions["save_scan"].setIcon(QIcon("icons/icons8-save1-50.png"))
        self.actions["save_multi"].setIcon(QIcon("icons/icons8-save2-50.png"))
        self.actions["fill"].setIcon(QIcon("icons/icons8-fill-color-50.png"))
        self.actions["flip"].setIcon(QIcon("icons/icons8-flip-48.png"))
        self.actions["zero"].setIcon(QIcon("icons/icons8-eyedropper-50.png"))
        self.actions["colormap"].setIcon(QIcon("icons/icons8-color-palette-50.png"))
        self.actions["compare"].setIcon(QIcon("icons/icons8-compare-50.png"))
        self.actions["profile"].setIcon(QIcon("icons/icons8-graph-50.png"))
        self.actions["about"].setIcon(QIcon("icons/icons8-about-50.png"))
        self.actions["exit"].setIcon(QIcon("icons/icons8-exit-50.png"))

        self.actions["colormap"].setCheckable(True)
        self.actions["colormap"].setChecked(False)

    def connect_actions(self):
        self.actions["open"].triggered.connect(self.open_file)
        self.actions["save_scan"].triggered.connect(self.save_single_scan)
        self.actions["save_multi"].triggered.connect(self.save_multiple_scans)
        self.actions["fill"].triggered.connect(self.fill_holes)
        self.actions["flip"].triggered.connect(self.flip_scan)
        self.actions["zero"].triggered.connect(self.set_zero_point_mode)
        self.actions["tilt"].triggered.connect(self.set_tilt_mode)
        self.actions["colormap"].triggered.connect(self.toggle_colormap_current_tab)
        self.actions["view3d"].triggered.connect(self.view3d)
        self.actions["compare"].triggered.connect(self.compare_scans)
        self.actions["profile"].triggered.connect(self.start_profile_analysis)
        self.actions["about"].triggered.connect(self.show_about_dialog)
        self.actions["exit"].triggered.connect(self.close)

        # connect_actions
        self.actions["del_outside"].triggered.connect(self.del_outside_mask)
        self.actions["del_inside"].triggered.connect(self.del_inside_mask)
        self.actions["show_mask"].triggered.connect(self.show_circle_roi)

    def create_menubar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        file_menu.addAction(self.actions["open"])
        file_menu.addAction(self.actions["save_scan"])
        file_menu.addAction(self.actions["save_multi"])

        self.recent_menu = QtWidgets.QMenu("Recent files", self)
        file_menu.addMenu(self.recent_menu)
        self.update_recent_files_menu()

        file_menu.addSeparator()
        file_menu.addAction(self.actions["exit"])

        edit_menu = menubar.addMenu("&Edit")        
        edit_menu.addAction(self.actions["show_mask"])
        delete_menu = QtWidgets.QMenu("delete", self)
        delete_menu.addAction(self.actions["del_outside"])
        delete_menu.addAction(self.actions["del_inside"])
        edit_menu.addMenu(delete_menu)


        actions_menu = menubar.addMenu("Scan &Actions")
        actions_menu.addAction(self.actions["fill"])
        actions_menu.addAction(self.actions["flip"])
        actions_menu.addAction(self.actions["zero"])
        actions_menu.addAction(self.actions["colormap"])

        tools_menu = menubar.addMenu("&Tools")
        tools_menu.addAction(self.actions["compare"])
        tools_menu.addAction(self.actions["profile"])

        help_menu = menubar.addMenu("&Help")
        help_menu.addAction(self.actions["about"])

    def create_toolbar(self):
        self.toolbar = self.addToolBar("Tools")
        self.toolbar.addAction(self.actions["open"])
        self.toolbar.addAction(self.actions["save_scan"])
        self.toolbar.addAction(self.actions["save_multi"])
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.actions["fill"])
        self.toolbar.addAction(self.actions["flip"])
        self.toolbar.addAction(self.actions["zero"])
        self.toolbar.addAction(self.actions["tilt"])
        self.toolbar.addAction(self.actions["colormap"])
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.actions["view3d"])
        self.toolbar.addAction(self.actions["compare"])
        self.toolbar.addAction(self.actions["profile"])
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.actions["about"])
        self.toolbar.addAction(self.actions["exit"])

    def create_circle_mask(self, shape, center, radius):
        Y, X = np.ogrid[:shape[0], :shape[1]]
        dist = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
        return dist <= radius


    def view3d(self):
        tab = self.current_tab()
        if tab:
            from simple3DWindow import Simple3DWindow
            if getattr(self, "_win3d", None) is None:
                self._win3d = Simple3DWindow(tab.grid)
            else:
                #self._win3d.hide()
                self._win3d.update_data(tab.grid)
            self._win3d.show()
            self._win3d.raise_()
            self._win3d.activateWindow()

    def toggle_colormap_current_tab(self):
        tab = self.current_tab()
        if tab:
            tab.toggle_colormap()

    def set_zero_point_mode(self):
        tab = self.current_tab()
        if tab:
            tab.set_zero_point_mode()

    def set_tilt_mode(self):
        tab = self.current_tab()
        if tab:
            tab.set_tilt_mode()

    def show_about_dialog(self):
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

    def load_csv(self, fname, tab):
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

    def load_npz(self, fname):
        data = np.load(fname)
        if 'frasta_info' in data:
            cnt = data['frasta_cnt']
            for i in range(cnt):
                try:
                    name = str(data[f"name_{i:02}"])
                    grid = data[f"grid_{i:02}"]
                    xi = data[f"xi_{i:02}"]
                    yi = data[f"yi_{i:02}"]
                    px_x = data[f"px_{i:02}"]
                    px_y = data[f"py_{i:02}"]

                    tab = ScanTab()
                    self.tabs.addTab(tab, str(name))
                    self.tabs.setCurrentWidget(tab)
                    tab.set_data(grid, xi, yi, px_x, px_y)
                except Exception as e:
                    QtWidgets.QMessageBox.critical(self, "Error", f"Error while loading:\n{e}")

            self.add_to_recent_files(fname)
            return True
        else:
            QtWidgets.QMessageBox.warning(self, "Format error", "NPZ does not contain a grid data.")
            # self.tabs.removeTab(self.tabs.indexOf(tab))
            return False

    def load_h5(self, fname):
        try:
            with h5py.File(fname, 'r') as f:
                if 'frasta_info' not in f.attrs:
                    QtWidgets.QMessageBox.warning(self, "Format error", "HDF5 does not contain a grid data.")
                    return False

                cnt = f.attrs.get('frasta_cnt', 0)
                for i in range(cnt):
                    try:
                        group_name = f"tab_{i:02}"
                        if group_name not in f:
                            continue

                        group = f[group_name]
                        name = group["name"][()].decode("utf-8")
                        grid = group["grid"][:]
                        xi = group["xi"][:]
                        yi = group["yi"][:]
                        px_x = group["px_x"][:]
                        px_y = group["px_y"][:]

                        tab = ScanTab()
                        self.tabs.addTab(tab, str(name))
                        self.tabs.setCurrentWidget(tab)
                        tab.set_data(grid, xi, yi, px_x, px_y)

                    except Exception as e:
                        QtWidgets.QMessageBox.critical(self, "Error", f"Error while loading tab {i}:\n{e}")

            self.add_to_recent_files(fname)
            return True

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error while opening HDF5 file:\n{e}")
            return False


    def create_tab_and_load(self, fname):
        if fname.endswith('.csv') or fname.endswith('.dat'):
            tab = ScanTab()
            self.tabs.addTab(tab, fname.split('/')[-1])
            self.tabs.setCurrentWidget(tab)
            self.load_csv(fname, tab)
            self.add_to_recent_files(fname)
        elif fname.endswith('.npz'):
            self.load_npz(fname)
        elif fname.endswith('.h5'):
            self.load_h5(fname)
        else:
            QtWidgets.QMessageBox.warning(self, "Unknown format", "Unsupported file type.")
            # self.tabs.removeTab(self.tabs.indexOf(tab))
            return

    def open_file(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open file", "", "CSV, NPZ, H5 (*.csv *.dat *.npz *.h5)")
        if not fname:
            return
        self.create_tab_and_load(fname)

    def open_file_from_recent(self, path):
        if not QtCore.QFile.exists(path):
            QtWidgets.QMessageBox.warning(self, "File not found", f"File not found:\n{path}")
            self.recent_files.remove(path)
            self.update_recent_files_menu()
            return
        self.create_tab_and_load(path)

    # format: tabs = [('name0', tab0), ('name1', tab1), ...]
    def save_tabs(self, tabs=None):
        if tabs is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "No data to save.")
            return

        fname, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Scan", "", "NPZ (*.npz);;HDF5 (*.h5)"
        )
        if not fname:
            return

        if selected_filter.startswith("NPZ") and not fname.endswith(".npz"):
            fname += ".npz"
        elif selected_filter.startswith("HDF5") and not fname.endswith(".h5"):
            fname += ".h5"

        i = 0
        try:
            if fname.endswith(".npz"):
                to_save = {
                    'frasta_info': "grid_data",
                }
                for name, tab in tabs:
                    to_save[f"name_{i:02}"] = name
                    to_save[f"grid_{i:02}"] = tab.grid
                    to_save[f"xi_{i:02}"] = tab.xi
                    to_save[f"yi_{i:02}"] = tab.yi
                    to_save[f"px_{i:02}"] = tab.px_x
                    to_save[f"py_{i:02}"] = tab.px_y
                    i += 1
                to_save['frasta_cnt'] = i

                np.savez_compressed(fname, **to_save)

            elif fname.endswith(".h5"):
                with h5py.File(fname, 'w') as f:
                    f.attrs['frasta_info'] = "grid_data"
                    for name, tab in tabs:
                        group = f.create_group(f"tab_{i:02}")
                        group.create_dataset("name", data=np.bytes_(name))
                        group.create_dataset("grid", data=tab.grid, compression="gzip", compression_opts=9)
                        group.create_dataset("xi", data=tab.xi, compression="gzip", compression_opts=9)
                        group.create_dataset("yi", data=tab.yi, compression="gzip", compression_opts=9)
                        group.create_dataset("px_x", data=np.atleast_1d(tab.px_x))
                        group.create_dataset("px_y", data=np.atleast_1d(tab.px_y))
                        i += 1
                    f.attrs['frasta_cnt'] = i

            QtWidgets.QMessageBox.information(self, "Saved", f"Scan saved to: {fname}")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error while saving:\n{e}")


    def save_single_scan(self):
        tab = self.current_tab()
        if not tab or not hasattr(tab, "grid") or tab.grid is None:
            QtWidgets.QMessageBox.warning(self, "No data", "No scan in current tab.")
            return

        self.save_tabs([("nowyskan", tab)])
        

    def save_multiple_scans(self):
        if self.tabs.count() == 0:
            QtWidgets.QMessageBox.warning(self, "No scans", "No scan tabs are open.")
            return

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Save selected scans")
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(QtWidgets.QLabel("Select scans to save and specify dataset names:"))

        checkboxes = []
        lineedits = []
        for i in range(self.tabs.count()):
            row = QtWidgets.QHBoxLayout()
            cb = QtWidgets.QCheckBox(self.tabs.tabText(i))
            cb.setChecked(True)
            le = QtWidgets.QLineEdit(self.tabs.tabText(i).replace(" ", "_"))
            row.addWidget(cb)
            row.addWidget(le)
            layout.addLayout(row)
            checkboxes.append(cb)
            lineedits.append(le)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        layout.addWidget(buttons)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return

        tabs = []
        for i, cb in enumerate(checkboxes):
            if cb.isChecked():
                dataset_name = lineedits[i].text().strip()
                if not dataset_name:
                    QtWidgets.QMessageBox.warning(self, "Invalid name", "Each scan must have a dataset name!")
                    return
                tab = self.tabs.widget(i)
                if not hasattr(tab, "grid") or tab.grid is None:
                    QtWidgets.QMessageBox.warning(self, "No data", f"Tab '{cb.text()}' has no scan data.")
                    return
                tabs.append((dataset_name, tab))

        if not tabs:
            QtWidgets.QMessageBox.warning(self, "Nothing to save", "No scans selected.")
            return

        self.save_tabs(tabs)


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

        def receive_aligned_grids(scan1_aligned_data : GridData, scan2_aligned_data : GridData, idx1=None, idx2=None):
            b = idx1 is not None and idx2 is not None
            if b:
                msg = QtWidgets.QMessageBox(self)
                msg.setWindowTitle("Dopasowanie skanów")
                msg.setText("Jak chcesz zapisać dopasowanie?")
                btn1 = msg.addButton("Jako nowe zakładki", QtWidgets.QMessageBox.AcceptRole)
                btn2 = msg.addButton("Nadpisz istniejące", QtWidgets.QMessageBox.ActionRole)
                msg.addButton("Anuluj", QtWidgets.QMessageBox.RejectRole)
                msg.exec_()

            if not b or msg.clickedButton() == btn1:
                tab1 = ScanTab()
                tab2 = ScanTab()
                self.tabs.addTab(tab1, "Dopasowany ref")
                self.tabs.addTab(tab2, "Dopasowany scan2")
            elif msg.clickedButton() == btn2:
                tab1 = self.tabs.widget(idx1)
                tab2 = self.tabs.widget(idx2)

            tab1.setGridData(scan1_aligned_data)
            tab2.setGridData(scan2_aligned_data)


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

        self.viewer = OverlayViewer( 
            tab1.getGridData(), 
            tab2.getGridData(),
            on_accept=partial(receive_aligned_grids, idx1=idx1, idx2=idx2)
        )

        self.viewer.setWindowTitle(f"Porównanie: {names[idx1]} vs {names[idx2]}")
        self.viewer.show()

    def start_profile_analysis(self):
        if self.tabs.count() < 2:
            QtWidgets.QMessageBox.warning(self, "Za mało skanów", "Musisz mieć co najmniej dwa skany!")
            return

        # Dialog wyboru dwóch zakładek
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Wybierz skany do analizy profilu")
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(QtWidgets.QLabel("Wybierz dwa skany:"))
        cb1 = QtWidgets.QComboBox()
        cb2 = QtWidgets.QComboBox()
        names = [self.tabs.tabText(i) for i in range(self.tabs.count())]
        cb1.addItems(names)
        cb2.addItems(names)
        layout.addWidget(QtWidgets.QLabel("Referencyjny skan:"))
        layout.addWidget(cb1)
        layout.addWidget(QtWidgets.QLabel("Skan do porównania:"))
        layout.addWidget(cb2)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        layout.addWidget(buttons)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return

        idx1 = cb1.currentIndex()
        idx2 = cb2.currentIndex()
        if idx1 == idx2:
            QtWidgets.QMessageBox.warning(self, "Błąd", "Wybierz dwa różne skany!")
            return

        tab1 = self.tabs.widget(idx1)
        tab2 = self.tabs.widget(idx2)

        # grid1 = tab1.grid
        # grid2 = tab2.grid
        grid1 = tab1.masked
        grid2 = tab2.masked

        # Sprawdzenie rozmiarów
        if grid1.shape != grid2.shape:
            # Minimalny wspólny rozmiar
            h = min(grid1.shape[0], grid2.shape[0])
            w = min(grid1.shape[1], grid2.shape[1])
            reply = QtWidgets.QMessageBox.question(
                self, "Różne rozmiary",
                f"Skany mają różne rozmiary:\n"
                f"{grid1.shape} vs {grid2.shape}\n"
                f"Przyciąć oba do wspólnego obszaru {h}x{w} i kontynuować?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return
            grid1 = grid1[:h, :w]
            grid2 = grid2[:h, :w]

        # viewer = ProfileViewer()
        # viewer.set_data(grid1, grid2, tab1.px_x, tab1.px_y, tab2.px_x, tab2.px_y)
        # viewer.show()
        # --- Uruchom ProfileViewer z przekazaniem danych ---
        # Możesz tu dynamicznie zaimportować profilViewer lub mieć własny wrapper:
        from profileViewer import ProfileViewer
        viewer = ProfileViewer()
        viewer.reference_grid = grid1
        viewer.adjusted_grid = grid2
        # wygładzone wersje na początek
        from scipy.ndimage import gaussian_filter
        sigma = 5.0  # możesz dodać pole wyboru
        viewer.reference_grid_smooth = gaussian_filter(grid1, sigma)
        viewer.adjusted_grid_smooth = gaussian_filter(grid2, sigma)
        # maska wspólna
        viewer.valid_mask = ~np.isnan(viewer.reference_grid_smooth) & ~np.isnan(viewer.adjusted_grid_smooth)
        # domyślna korekcja
        viewer.adjusted_grid_corrected = viewer.adjusted_grid_smooth + np.nanmean(viewer.reference_grid_smooth - viewer.adjusted_grid_smooth)
        # skopiuj też metadane, jeśli masz w ScanTab
        if hasattr(self.tabs.widget(idx1), "metadata"):
            viewer.metadata = self.tabs.widget(idx1).metadata
        # dokończ GUI i pokaż
        viewer.show()
        viewer.on_worker_finished({
            "reference_grid": grid1,
            "adjusted_grid": grid2,
            "reference_grid_smooth": viewer.reference_grid_smooth,
            "adjusted_grid_smooth": viewer.adjusted_grid_smooth,
            "valid_mask": viewer.valid_mask,
            "adjusted_grid_corrected": viewer.adjusted_grid_corrected,
        })


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
