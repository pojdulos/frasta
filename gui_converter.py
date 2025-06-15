import sys
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QIcon
from overlayViewer import OverlayViewer
from aboutDialog import AboutDialog
from scanTab import ScanTab


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

        save_scan_action = QtWidgets.QAction("Save current scan as NPY...", self)
        save_scan_action.triggered.connect(self.save_single_scan)

        save_multi_action = QtWidgets.QAction("Save multiple scans...", self)
        save_multi_action.triggered.connect(self.save_multiple_scans)

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

        profile_action = QtWidgets.QAction("Profile analysis...", self)
        profile_action.triggered.connect(self.start_profile_analysis)

        open_action.setIcon(QIcon("icons/icons8-open-file1-50.png"))
        save_scan_action.setIcon(QIcon("icons/icons8-save1-50.png"))
        save_multi_action.setIcon(QIcon("icons/icons8-save2-50.png"))
        fill_action.setIcon(QIcon("icons/icons8-fill-color-50.png"))
        flip_action.setIcon(QIcon("icons/icons8-flip-48.png"))
        compare_action.setIcon(QIcon("icons/icons8-compare-50.png"))
        zero_action.setIcon(QIcon("icons/icons8-eyedropper-50.png"))
        about_action.setIcon(QIcon("icons/icons8-about-50.png"))
        exit_action.setIcon(QIcon("icons/icons8-exit-50.png"))
        colormap_action.setIcon(QIcon("icons/icons8-color-palette-50.png"))
        profile_action.setIcon(QIcon("icons/icons8-graph-50.png"))


        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        file_menu.addAction(open_action)
        file_menu.addAction(save_scan_action)
        file_menu.addAction(save_multi_action)

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
        self.toolbar.addAction(save_scan_action)
        self.toolbar.addAction(save_multi_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(fill_action)
        self.toolbar.addAction(flip_action)
        self.toolbar.addAction(zero_action)
        self.toolbar.addAction(colormap_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(compare_action)
        self.toolbar.addAction(profile_action)
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

    def save_single_scan(self):
        tab = self.current_tab()
        if not tab or not hasattr(tab, "grid") or tab.grid is None:
            QtWidgets.QMessageBox.warning(self, "No data", "No scan in current tab.")
            return
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save as NPY", "", "NPY (*.npy)")
        if not fname:
            return
        try:
            np.save(fname, tab.grid)
            QtWidgets.QMessageBox.information(self, "Saved", f"Scan saved to: {fname}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error while saving:\n{e}")

    def save_multiple_scans(self):
        if self.tabs.count() == 0:
            QtWidgets.QMessageBox.warning(self, "No scans", "No scan tabs are open.")
            return

        # --- 1. Dialog wyboru skanów i nazw ---
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

        layout.addWidget(QtWidgets.QLabel("Select file format:"))
        format_combo = QtWidgets.QComboBox()
        format_combo.addItems(["NPZ (compressed)", "HDF5"])
        layout.addWidget(format_combo)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        layout.addWidget(buttons)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return

        # --- 2. Zbieranie wyboru ---
        to_save = []
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
                to_save.append((dataset_name, tab.grid))

        if not to_save:
            QtWidgets.QMessageBox.warning(self, "Nothing to save", "No scans selected.")
            return

        # --- 3. Zapytaj o plik docelowy ---
        fmt = format_combo.currentText()
        if fmt.startswith("NPZ"):
            fname, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save as NPZ", "", "NPZ (*.npz)")
            if not fname:
                return
            try:
                np.savez_compressed(fname, **{name: grid for name, grid in to_save})
                QtWidgets.QMessageBox.information(self, "Saved", f"Scans saved to: {fname}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Error while saving:\n{e}")

        else:  # HDF5
            fname, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save as HDF5", "", "HDF5 (*.h5)")
            if not fname:
                return
            import h5py
            try:
                with h5py.File(fname, "a") as f:
                    for name, grid in to_save:
                        if name in f:
                            msg = QtWidgets.QMessageBox.question(
                                self, "Dataset exists",
                                f"Dataset '{name}' already exists.\nOverwrite?",
                                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
                            )
                            if msg != QtWidgets.QMessageBox.Yes:
                                continue
                            del f[name]
                        f.create_dataset(name, data=grid)
                QtWidgets.QMessageBox.information(self, "Saved", f"Scans saved to: {fname}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Error while saving:\n{e}")

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

        def receive_aligned_grids(scan1_aligned, scan2_aligned):
            print(f"scan1: {scan1_aligned.shape}, scan2: {scan2_aligned.shape}")
            # popup: nadpisać czy dodać nowe zakładki?
            msg = QtWidgets.QMessageBox(self)
            msg.setWindowTitle("Dopasowanie skanów")
            msg.setText("Jak chcesz zapisać dopasowanie?")
            btn1 = msg.addButton("Jako nowe zakładki", QtWidgets.QMessageBox.AcceptRole)
            btn2 = msg.addButton("Nadpisz istniejące", QtWidgets.QMessageBox.ActionRole)
            msg.addButton("Anuluj", QtWidgets.QMessageBox.RejectRole)
            msg.exec_()
            if msg.clickedButton() == btn1:
                tab1 = ScanTab()
                tab2 = ScanTab()
                tab1.set_data(scan1_aligned, tab1.xi, tab1.yi, tab1.px_x, tab1.px_y, None, None, None)
                tab2.set_data(scan2_aligned, tab2.xi, tab2.yi, tab2.px_x, tab2.px_y, None, None, None)
                self.tabs.addTab(tab1, "Dopasowany ref")
                self.tabs.addTab(tab2, "Dopasowany scan2")
            elif msg.clickedButton() == btn2:
                current_tab = self.tabs.currentWidget()
                if isinstance(current_tab, ScanTab):
                    current_tab.set_data(scan2_aligned, current_tab.xi, current_tab.yi, current_tab.px_x, current_tab.px_y, None, None, None)
            # jeśli "Anuluj", nie rób nic

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
        self.viewer = OverlayViewer(grid1, grid2, on_accept=receive_aligned_grids)
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

        grid1 = self.tabs.widget(idx1).grid
        grid2 = self.tabs.widget(idx2).grid

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
