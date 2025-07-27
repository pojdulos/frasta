from pyqtgraph.Qt import QtWidgets, QtGui, QtCore
import numpy as np
import pyqtgraph.opengl as gl


class Grid3DViewer(QtWidgets.QWidget):
    def __init__(self, reference_grid, adjusted_grid=None, line_points=None,
                 separation=0, ref_surface_mode='surface', show_controls=True, parent=None):
        super().__init__(parent)
        self.setWindowTitle("3D Grid Viewer")
        self.resize(900, 700)
        layout = QtWidgets.QVBoxLayout(self)

        self.show_controls = show_controls
        self.ref_surface_mode = ref_surface_mode

        if show_controls:
            self.init_controls(layout)

        self.view = gl.GLViewWidget()
        layout.addWidget(self.view)
        self.view.setCameraPosition(distance=200)

        self.surface_ref_item = None
        self.surface_adj_item = None
        self.ref_profile_line_item = None
        self.adj_profile_line_item = None
        self.cross_plane_item = None

        self.update_data(reference_grid, adjusted_grid, line_points, separation)

        if show_controls:
            self.connect_controls()

    def init_controls(self, layout):
        ctrl_layout = QtWidgets.QHBoxLayout()
        self.checkbox_ref = QtWidgets.QCheckBox("Show Reference Surface")
        self.checkbox_ref.setChecked(True)
        self.checkbox_adj = QtWidgets.QCheckBox("Show Adjusted Surface")
        self.checkbox_adj.setChecked(True)
        self.checkbox_line = QtWidgets.QCheckBox("Show Profile Line")
        self.checkbox_line.setChecked(True)
        self.checkbox_plane = QtWidgets.QCheckBox("Show Section Plane")
        self.checkbox_plane.setChecked(True)
        for cb in [self.checkbox_ref, self.checkbox_adj, self.checkbox_line, self.checkbox_plane]:
            ctrl_layout.addWidget(cb)
        layout.addLayout(ctrl_layout)

    def connect_controls(self):
        self.checkbox_ref.stateChanged.connect(self.toggle_surface_ref)
        self.checkbox_adj.stateChanged.connect(self.toggle_surface_adj)
        self.checkbox_line.stateChanged.connect(self.toggle_profile_line)
        self.checkbox_plane.stateChanged.connect(self.toggle_cross_plane)

    def update_data(self, reference_grid, adjusted_grid=None, line_points=None, separation=0):
        for item in [self.surface_ref_item, self.surface_adj_item,
                     self.ref_profile_line_item, self.adj_profile_line_item,
                     self.cross_plane_item]:
            if item:
                self.view.removeItem(item)

        self.surface_ref_item = None
        self.surface_adj_item = None
        self.ref_profile_line_item = None
        self.adj_profile_line_item = None
        self.cross_plane_item = None

        step = max(1, min(reference_grid.shape[0], reference_grid.shape[1]) // 512)
        ys = np.arange(0, reference_grid.shape[0], step)
        xs = np.arange(0, reference_grid.shape[1], step)
        Z_ref = reference_grid[np.ix_(ys, xs)]
        Z_ref = np.where(np.isfinite(Z_ref), Z_ref, np.nan)
        Z_ref = np.where((Z_ref > 1e6) | (Z_ref < -1e6), 0.0, Z_ref)

        if Z_ref.shape == (len(ys), len(xs)) and not np.all(np.isnan(Z_ref)):
            if self.ref_surface_mode == 'mesh':
                self.surface_ref_item = self.make_voxel_mesh(Z_ref)
            else:
                self.surface_ref_item = gl.GLSurfacePlotItem(
                    x=xs, y=ys, z=Z_ref.T, color=(0,1,0,1), shader='shaded')
            self.view.addItem(self.surface_ref_item)

        if adjusted_grid is not None:
            Z_adj = adjusted_grid[np.ix_(ys, xs)] + separation
            Z_adj = np.where(np.isfinite(Z_adj), Z_adj, np.nan)
            Z_adj = np.where((Z_adj > 1e6) | (Z_adj < -1e6), 0.0, Z_adj)
            if Z_adj.shape == (len(ys), len(xs)) and not np.all(np.isnan(Z_adj)):
                self.surface_adj_item = gl.GLSurfacePlotItem(
                    x=xs, y=ys, z=Z_adj.T, color=(0.2,0.3,1,1), shader='shaded')
                self.view.addItem(self.surface_adj_item)
        else:
            Z_adj = np.full_like(Z_ref, np.nan)

        if not np.any(np.isfinite(Z_ref)) and not np.any(np.isfinite(Z_adj)):
            return  # nothing to display safely

        if adjusted_grid is not None:
            z_min = min(np.nanmin(Z_ref), np.nanmin(Z_adj))
            z_max = max(np.nanmax(Z_ref), np.nanmax(Z_adj))
        else:
            z_min = np.nanmin(Z_ref)
            z_max = np.nanmax(Z_ref)

        margin = 0.1 * (z_max - z_min)
        z_min -= margin
        z_max += margin

        if line_points is not None and len(line_points) > 1:
            pts = np.array(line_points)
            self.cross_plane_item = self.add_cross_section_plane(pts, z_min, z_max)
            try:
                ref_prof = reference_grid[pts[:,1], pts[:,0]]
                self.ref_profile_line_item = gl.GLLinePlotItem(
                    pos=np.column_stack((pts[:,0], pts[:,1], ref_prof)),
                    color=(0,0.4,0,1), width=2)
                self.view.addItem(self.ref_profile_line_item)
                if adjusted_grid is not None:
                    adj_prof = adjusted_grid[pts[:,1], pts[:,0]] + separation
                    self.adj_profile_line_item = gl.GLLinePlotItem(
                        pos=np.column_stack((pts[:,0], pts[:,1], adj_prof)),
                        color=(0,0,1,1), width=2)
                    self.view.addItem(self.adj_profile_line_item)
            except Exception as e:
                print("[Grid3DViewer] Failed to plot profile lines:", e)

        all_x = list(xs)
        all_y = list(ys)
        all_z = [np.nanmin(Z_ref), np.nanmax(Z_ref)]
        if not np.all(np.isnan(Z_adj)):
            all_z += [np.nanmin(Z_adj), np.nanmax(Z_adj)]
        if line_points is not None:
            pts = np.array(line_points)
            all_x += list(pts[:,0])
            all_y += list(pts[:,1])

        xc = (min(all_x) + max(all_x)) / 2
        yc = (min(all_y) + max(all_y)) / 2
        zc = (min(all_z) + max(all_z)) / 2 if all_z else 0
        self.view.setCameraPosition(pos=QtGui.QVector3D(xc, yc, zc))

    def make_voxel_mesh(self, Z, pixel_size=1.0):
        rows, cols = Z.shape
        verts, faces, colors = [], [], []

        def add_quad(v0, v1, v2, v3, color):
            idx = len(verts)
            verts.extend([v0, v1, v2, v3])
            faces.extend([[idx, idx+1, idx+2], [idx, idx+2, idx+3]])
            colors.extend([color]*2)

        for i in range(rows):
            for j in range(cols):
                z = Z[i, j]
                if np.isnan(z):
                    continue
                x0, x1 = j*pixel_size, (j+1)*pixel_size
                y0, y1 = i*pixel_size, (i+1)*pixel_size
                v0, v1 = [x0, y0, z], [x1, y0, z]
                v2, v3 = [x1, y1, z], [x0, y1, z]
                add_quad(v0, v1, v2, v3, [0.7, 0.7, 0.7, 1])

                if j < cols-1 and not np.isnan(Z[i, j+1]) and Z[i, j+1] != z:
                    z2 = Z[i, j+1]
                    add_quad([x1, y0, z], [x1, y0, z2], [x1, y1, z2], [x1, y1, z], [0.5, 0.5, 1, 1])
                if i < rows-1 and not np.isnan(Z[i+1, j]) and Z[i+1, j] != z:
                    z2 = Z[i+1, j]
                    add_quad([x0, y1, z], [x1, y1, z], [x1, y1, z2], [x0, y1, z2], [0.5, 0.5, 1, 1])

        return gl.GLMeshItem(vertexes=np.array(verts), faces=np.array(faces),
                             faceColors=np.array(colors), smooth=False, drawEdges=False)

    def add_cross_section_plane(self, pts, z_min, z_max):
        p0, p1 = pts[0], pts[-1]
        rect = np.array([
            [p0[0], p0[1], z_min],
            [p1[0], p1[1], z_min],
            [p1[0], p1[1], z_max],
            [p0[0], p0[1], z_max],
        ])
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        color = np.array([0.5, 0.5, 0.7, 0.7])
        mesh = gl.GLMeshItem(vertexes=rect, faces=faces, faceColors=np.tile(color, (2,1)),
                             glOptions='translucent', smooth=False, drawEdges=False)
        self.view.addItem(mesh)
        return mesh

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

    def closeEvent(self, event):
        # Usuń wszystkie elementy z widoku
        for item in [self.surface_ref_item, self.surface_adj_item,
                    self.ref_profile_line_item, self.adj_profile_line_item,
                    self.cross_plane_item]:
            if item:
                self.view.removeItem(item)

        self.surface_ref_item = None
        self.surface_adj_item = None
        self.ref_profile_line_item = None
        self.adj_profile_line_item = None
        self.cross_plane_item = None

        # Wymuś natychmiastowe przerysowanie pustej sceny
        self.view.repaint()          # odświeżenie QWidgetu
        QtWidgets.QApplication.processEvents()  # przetwórz natychmiast

        event.accept()



_global_3d_viewer = None

def show_3d_viewer(reference_grid, adjusted_grid=None, line_points=None, separation=0, show_controls=True):
    global _global_3d_viewer
    if _global_3d_viewer is None:
        _global_3d_viewer = Grid3DViewer(
            reference_grid=reference_grid,
            adjusted_grid=adjusted_grid,
            line_points=line_points,
            separation=separation,
            show_controls=show_controls
        )
    else:
        _global_3d_viewer.update_data(
            reference_grid=reference_grid,
            adjusted_grid=adjusted_grid,
            line_points=line_points,
            separation=separation
        )
    _global_3d_viewer.show()
    _global_3d_viewer.raise_()
    _global_3d_viewer.activateWindow()

