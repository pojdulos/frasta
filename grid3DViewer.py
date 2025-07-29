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

        self.ref_surface_mode = ref_surface_mode

        self.init_controls(layout)
        self.connect_controls()

        self.view = gl.GLViewWidget()
        layout.addWidget(self.view)
        self.view.setCameraPosition(distance=200)

        self.surface_ref_item = None
        self.surface_adj_item = None
        self.ref_profile_line_item = None
        self.adj_profile_line_item = None
        self.cross_plane_item = None

        self.set_controls_visible(show_controls)
        self.update_data(reference_grid, adjusted_grid=adjusted_grid, line_points=line_points, separation=separation)

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

    def remove_existing_items(self):
        """Removes all existing items from the 3D view.

        Clears reference and adjusted surfaces, profile lines, and cross-section planes from the viewer.
        """
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

    def update_data(self, reference_grid, adjusted_grid=None, line_points=None, separation=0):
        """Updates the 3D view with new grid data and profile lines.

        Removes existing items, prepares and adds new surfaces and profile lines, and recenters the camera.
        
        Args:
            reference_grid (np.ndarray): The reference grid data.
            adjusted_grid (np.ndarray, optional): The adjusted grid data.
            line_points (list or np.ndarray, optional): Points for the profile line.
            separation (float, optional): Vertical separation between surfaces.
        """
        self.remove_existing_items()

        xs, ys, Z_ref = self._prepare_reference_surface(reference_grid)
        Z_adj = self._prepare_adjusted_surface(adjusted_grid, ys, xs, separation, Z_ref)

        self._add_reference_surface(xs, ys, Z_ref)
        self._add_adjusted_surface(xs, ys, Z_adj)

        if not np.any(np.isfinite(Z_ref)) and not np.any(np.isfinite(Z_adj)):
            return  # nothing to display safely

        z_min, z_max = self._compute_z_limits(Z_ref, Z_adj, adjusted_grid is not None)
        margin = 0.1 * (z_max - z_min)
        z_min -= margin
        z_max += margin

        self._add_profile_and_plane(reference_grid, adjusted_grid, line_points, separation, z_min, z_max)

        self._center_camera(xs, ys, Z_ref, Z_adj, line_points)

    def _prepare_reference_surface(self, reference_grid):
        """Prepares the reference surface for 3D visualization.

        Downsamples the reference grid, replaces invalid values with NaN, and returns the axes and processed grid.

        Args:
            reference_grid (np.ndarray): The reference grid data.

        Returns:
            tuple: (xs, ys, Z_ref) where xs and ys are axis arrays and Z_ref is the processed grid.
        """
        max_points = 256
        step = max(1, min(reference_grid.shape[0], reference_grid.shape[1]) // max_points)
        ys = np.arange(0, reference_grid.shape[0], step)
        xs = np.arange(0, reference_grid.shape[1], step)
        Z_ref = reference_grid[np.ix_(ys, xs)]
        Z_ref = np.where(np.isfinite(Z_ref), Z_ref, np.nan)
        Z_ref = np.where((Z_ref > 1e6) | (Z_ref < -1e6), np.nan, Z_ref)
        return xs, ys, Z_ref

    def _prepare_adjusted_surface(self, adjusted_grid, ys, xs, separation, Z_ref):
        """Prepares the adjusted surface for 3D visualization.

        Processes the adjusted grid by downsampling, applying separation, and replacing invalid values with NaN.

        Args:
            adjusted_grid (np.ndarray or None): The adjusted grid data.
            ys (np.ndarray): Y-axis indices for downsampling.
            xs (np.ndarray): X-axis indices for downsampling.
            separation (float): Vertical separation to apply.
            Z_ref (np.ndarray): Reference grid for shape if adjusted_grid is None.

        Returns:
            np.ndarray: The processed adjusted grid.
        """
        if adjusted_grid is not None:
            Z_adj = adjusted_grid[np.ix_(ys, xs)] + separation
            Z_adj = np.where(np.isfinite(Z_adj), Z_adj, np.nan)
            Z_adj = np.where((Z_adj > 1e6) | (Z_adj < -1e6), np.nan, Z_adj)
        else:
            Z_adj = np.full_like(Z_ref, np.nan)
        return Z_adj

    def _add_reference_surface(self, xs, ys, Z_ref):
        """Adds the reference surface to the 3D view if valid data is present.

        Depending on the surface mode, creates either a voxel mesh, a shaded surface plot, or a wireframe and adds it to the view.

        Args:
            xs (np.ndarray): X-axis indices.
            ys (np.ndarray): Y-axis indices.
            Z_ref (np.ndarray): Processed reference grid.
        """
        if Z_ref.shape == (len(ys), len(xs)) and not np.all(np.isnan(Z_ref)):
            if self.ref_surface_mode == 'mesh':
                self.surface_ref_item = self.make_voxel_mesh(Z_ref)
            elif self.ref_surface_mode == 'wireframe':
                self.surface_ref_item = gl.GLSurfacePlotItem(
                    x=xs, y=ys, z=Z_ref.T, color=(0,1,0,1), drawFaces=False, drawEdges=True, edgeColor=(0,1,0,1))
            else:
                self.surface_ref_item = gl.GLSurfacePlotItem(
                    x=xs, y=ys, z=Z_ref.T, color=(0,1,0,1), shader='shaded')
            self.view.addItem(self.surface_ref_item)

    def _add_adjusted_surface(self, xs, ys, Z_adj):
        """Adds the adjusted surface to the 3D view if valid data is present.

        Creates a shaded surface plot or wireframe for the adjusted grid and adds it to the view.

        Args:
            xs (np.ndarray): X-axis indices.
            ys (np.ndarray): Y-axis indices.
            Z_adj (np.ndarray): Processed adjusted grid.
        """
        if Z_adj.shape == (len(ys), len(xs)) and not np.all(np.isnan(Z_adj)):
            if self.ref_surface_mode == 'wireframe':
                self.surface_adj_item = gl.GLSurfacePlotItem(
                    x=xs, y=ys, z=Z_adj.T, color=(0.2,0.3,1,1), drawFaces=False, drawEdges=True, edgeColor=(0.2,0.3,1,1))
            else:
                self.surface_adj_item = gl.GLSurfacePlotItem(
                    x=xs, y=ys, z=Z_adj.T, color=(0.2,0.3,1,1), shader='shaded')
            self.view.addItem(self.surface_adj_item)

    def _compute_z_limits(self, Z_ref, Z_adj, has_adjusted):
        """Computes the minimum and maximum Z values for the 3D view.

        Determines the Z-axis limits based on the reference and adjusted grids.

        Args:
            Z_ref (np.ndarray): Processed reference grid.
            Z_adj (np.ndarray): Processed adjusted grid.
            has_adjusted (bool): Whether the adjusted grid is present.

        Returns:
            tuple: (z_min, z_max) representing the Z-axis limits.
        """
        if has_adjusted:
            z_min = min(np.nanmin(Z_ref), np.nanmin(Z_adj))
            z_max = max(np.nanmax(Z_ref), np.nanmax(Z_adj))
        else:
            z_min = np.nanmin(Z_ref)
            z_max = np.nanmax(Z_ref)
        return z_min, z_max

    def _add_profile_and_plane(self, reference_grid, adjusted_grid, line_points, separation, z_min, z_max):
        """Adds a cross-section plane and profile lines to the 3D view if line points are provided.

        Draws the cross-section plane and profile lines for both reference and adjusted grids, if available.

        Args:
            reference_grid (np.ndarray): The reference grid data.
            adjusted_grid (np.ndarray or None): The adjusted grid data.
            line_points (list or np.ndarray): Points for the profile line.
            separation (float): Vertical separation between surfaces.
            z_min (float): Minimum Z value for the plane.
            z_max (float): Maximum Z value for the plane.
        """
        if line_points is not None and len(line_points) > 1:
            pts = np.array(line_points)
            self.cross_plane_item = self.add_cross_section_plane(pts, z_min, z_max)
            h, w = reference_grid.shape
            valid_mask = (
                (pts[:, 1] >= 0) & (pts[:, 1] < h) &
                (pts[:, 0] >= 0) & (pts[:, 0] < w)
            )
            if not np.all(valid_mask):
                print("[Grid3DViewer] Some profile points are out of bounds and will be ignored.")
                pts = pts[valid_mask]
                if len(pts) < 2:
                    return  # not enough points to plot a line
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



    def _center_camera(self, xs, ys, Z_ref, Z_adj, line_points):
        """Recenters the 3D camera based on the current data bounds.

        Calculates the center of all displayed data and sets the camera position accordingly.

        Args:
            xs (np.ndarray): X-axis indices.
            ys (np.ndarray): Y-axis indices.
            Z_ref (np.ndarray): Processed reference grid.
            Z_adj (np.ndarray): Processed adjusted grid.
            line_points (list or np.ndarray): Points for the profile line.
        """
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

    def set_controls_visible(self, visible):
        self.show_controls = visible
        if hasattr(self, "checkbox_ref"):
            for cb in [self.checkbox_ref, self.checkbox_adj, self.checkbox_line, self.checkbox_plane]:
                cb.setVisible(visible)
            self.update()

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
        self.remove_existing_items()
        self.view.repaint()          # odświeżenie QWidgetu
        QtWidgets.QApplication.processEvents()  # przetwórz natychmiast
        event.accept()



_global_3d_viewer = None

def show_3d_viewer(reference_grid, adjusted_grid=None, line_points=None, separation=0, show_controls=True):
    """Displays a 3D viewer window for visualizing reference and adjusted grids.

    Opens or updates a global 3D viewer with the provided grid data, profile lines, and display options.

    Args:
        reference_grid (np.ndarray): The reference grid data to display.
        adjusted_grid (np.ndarray, optional): The adjusted grid data to display.
        line_points (list or np.ndarray, optional): Points for the profile line.
        separation (float, optional): Vertical separation between surfaces.
        show_controls (bool, optional): Whether to show control checkboxes in the viewer.
    """
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
        _global_3d_viewer.set_controls_visible(show_controls)
        _global_3d_viewer.update_data(
            reference_grid=reference_grid,
            adjusted_grid=adjusted_grid,
            line_points=line_points,
            separation=separation
        )
    _global_3d_viewer.show()
    _global_3d_viewer.raise_()
    _global_3d_viewer.activateWindow()


if __name__ == '__main__':
    from frasta_gui import run
    run()
