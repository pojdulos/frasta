from pyqtgraph.Qt import QtWidgets, QtGui, QtCore
import numpy as np
import pyqtgraph.opengl as gl


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


        step = max(1, min(reference_grid.shape[0], reference_grid.shape[1]) // 512)
        #step = 5

        print(f"step={step}")        
        ys = np.arange(0, reference_grid.shape[0], step)
        xs = np.arange(0, reference_grid.shape[1], step)
        Z_ref = reference_grid[np.ix_(ys, xs)]
        Z_adj = adjusted_grid[np.ix_(ys, xs)] + separation

        print('Z_ref min:', np.nanmin(Z_ref), 'Z_ref max:', np.nanmax(Z_ref), 'ref shape:', Z_ref.shape)
        print('Z_ref NaN count:', np.isnan(Z_ref).sum())

        print('Z_adj min:', np.nanmin(Z_adj), 'adj max:', np.nanmax(Z_adj), 'adj shape:', Z_adj.shape)
        print('Z_adj NaN count:', np.isnan(Z_adj).sum())

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

        # Z_ref = Z_ref 
        # Z_adj = Z_adj

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

        z_min -= np.abs(0.1 * z_min)
        z_max += np.abs(0.1 * z_max)

        self.cross_plane_item = self.add_cross_section_plane(line_points, z_min, z_max)

        # Dodaj linię profilu
        if line_points is not None and len(line_points) > 1:
            pts = np.array(line_points)
            ref_prof = reference_grid[pts[:,1], pts[:,0]]
            ref_prof = ref_prof #/ 1000.0

            self.ref_profile_line_item = gl.GLLinePlotItem(
                pos=np.column_stack((pts[:,0], pts[:,1], ref_prof)),
                color=(0,0.4,0,1), width=2
                #, glOptions='opaque'
                #, antialias=True
            )
            self.view.addItem(self.ref_profile_line_item)

            adj_prof = adjusted_grid[pts[:,1], pts[:,0]] + separation
            adj_prof = adj_prof #/ 1000.0

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

        # Wyznacz bounding box
        all_x = []
        all_y = []
        all_z = []

        if not np.all(np.isnan(Z_ref)):
            all_x.extend(xs)
            all_y.extend(ys)
            all_z.append(np.nanmin(Z_ref))
            all_z.append(np.nanmax(Z_ref))

        if not np.all(np.isnan(Z_adj)):
            all_x.extend(xs)
            all_y.extend(ys)
            all_z.append(np.nanmin(Z_adj))
            all_z.append(np.nanmax(Z_adj))

        # Dołóż jeszcze punkty z profilu jeśli są
        if line_points is not None and len(line_points) > 1:
            pts = np.array(line_points)
            all_x.extend(pts[:,0])
            all_y.extend(pts[:,1])
            # Możesz też dołożyć z profilu Z jeśli chcesz pełny bbox

        # Licz środek sceny
        xc = (min(all_x) + max(all_x)) / 2
        yc = (min(all_y) + max(all_y)) / 2
        zc = (min(all_z) + max(all_z)) / 2 if len(all_z) > 0 else 0

        self.view.setCameraPosition(pos=QtGui.QVector3D(xc, yc, zc))



    def update_data(self, reference_grid, adjusted_grid, line_points=None, separation=0):
        """
        Aktualizuje dane w oknie 3D: czyści stare elementy i rysuje nowe na podstawie przekazanych danych.
        """
        self.view.clear()
        self.surface_ref_item = None
        self.surface_adj_item = None
        self.ref_profile_line_item = None
        self.adj_profile_line_item = None
        self.cross_plane_item = None

        step = max(1, min(reference_grid.shape[0], reference_grid.shape[1]) // 512)
        ys = np.arange(0, reference_grid.shape[0], step)
        xs = np.arange(0, reference_grid.shape[1], step)
        Z_ref = reference_grid[np.ix_(ys, xs)]
        Z_adj = adjusted_grid[np.ix_(ys, xs)] + separation

        Z_MAX = 1e6
        Z_MIN = -1e6
        Z_ref = np.where(np.isfinite(Z_ref), Z_ref, np.nan)
        Z_adj = np.where(np.isfinite(Z_adj), Z_adj, np.nan)
        Z_ref = np.where((Z_ref > Z_MAX) | (Z_ref < Z_MIN), 0.0, Z_ref)
        Z_adj = np.where((Z_adj > Z_MAX) | (Z_adj < Z_MIN), 0.0, Z_adj)

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
        z_min -= np.abs(0.1 * z_min)
        z_max += np.abs(0.1 * z_max)

        self.cross_plane_item = self.add_cross_section_plane(line_points, z_min, z_max)

        if line_points is not None and len(line_points) > 1:
            pts = np.array(line_points)
            ref_prof = reference_grid[pts[:,1], pts[:,0]]

            self.ref_profile_line_item = gl.GLLinePlotItem(
                pos=np.column_stack((pts[:,0], pts[:,1], ref_prof)),
                color=(0,0.4,0,1), width=2
            )
            self.view.addItem(self.ref_profile_line_item)

            adj_prof = adjusted_grid[pts[:,1], pts[:,0]] + separation
            self.adj_profile_line_item = gl.GLLinePlotItem(
                pos=np.column_stack((pts[:,0], pts[:,1], adj_prof)),
                color=(0,0,1,1), width=2
            )
            self.view.addItem(self.adj_profile_line_item)

        # Ponownie połącz checkboxy z widocznością nowych elementów
        self.checkbox_ref.stateChanged.connect(self.toggle_surface_ref)
        self.checkbox_adj.stateChanged.connect(self.toggle_surface_adj)
        self.checkbox_line.stateChanged.connect(self.toggle_profile_line)
        self.checkbox_plane.stateChanged.connect(self.toggle_cross_plane)

        # Ustaw kamerę na środek nowych danych
        all_x = []
        all_y = []
        all_z = []
        if not np.all(np.isnan(Z_ref)):
            all_x.extend(xs)
            all_y.extend(ys)
            all_z.append(np.nanmin(Z_ref))
            all_z.append(np.nanmax(Z_ref))
        if not np.all(np.isnan(Z_adj)):
            all_x.extend(xs)
            all_y.extend(ys)
            all_z.append(np.nanmin(Z_adj))
            all_z.append(np.nanmax(Z_adj))
        if line_points is not None and len(line_points) > 1:
            pts = np.array(line_points)
            all_x.extend(pts[:,0])
            all_y.extend(pts[:,1])
        xc = (min(all_x) + max(all_x)) / 2
        yc = (min(all_y) + max(all_y)) / 2
        zc = (min(all_z) + max(all_z)) / 2 if len(all_z) > 0 else 0

        self.view.setCameraPosition(pos=QtGui.QVector3D(xc, yc, zc))


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




from pyqtgraph.Qt import QtWidgets, QtGui, QtCore
import numpy as np
import pyqtgraph.opengl as gl

class Profile3DWindowTestMesh(QtWidgets.QWidget):
    def __init__(self, reference_grid, adjusted_grid, line_points=None, separation=0, auto_center_z=True, parent=None):
        super().__init__(parent)
        self.setWindowTitle("3D View of Grids and Profile Plane")
        self.resize(900, 700)
        layout = QtWidgets.QVBoxLayout(self)

        # ---- Panel z checkboxami ----
        ctrl_layout = QtWidgets.QHBoxLayout()
        self.checkbox_ref = QtWidgets.QCheckBox("Show Reference Surface (stairs)")
        self.checkbox_ref.setChecked(True)
        self.checkbox_adj = QtWidgets.QCheckBox("Show Adjusted Surface (smooth)")
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

        # Mesh (stairs) – dla powierzchni referencyjnej
        step = max(1, min(reference_grid.shape[0], reference_grid.shape[1]) // 2048)
        ys = np.arange(0, reference_grid.shape[0], step)
        xs = np.arange(0, reference_grid.shape[1], step)
        Z_ref = reference_grid[np.ix_(ys, xs)]

        self.surface_ref_item = self.make_voxel_mesh(Z_ref)
        self.view.addItem(self.surface_ref_item)

        # Powierzchnia dopasowana – nadal jako GLSurfacePlotItem
        Z_adj = adjusted_grid[np.ix_(ys, xs)] + separation
        Z_adj = np.where(np.isfinite(Z_adj), Z_adj, np.nan)
        Z_adj = np.where((Z_adj > 1e6) | (Z_adj < -1e6), 0.0, Z_adj)

        if not np.all(np.isnan(Z_adj)):
            self.surface_adj_item = gl.GLSurfacePlotItem(
                x=xs, y=ys, z=Z_adj.T, color=(0.2,0.3,1,1), shader='shaded'
            )
            self.view.addItem(self.surface_adj_item)

        # Płaszczyzna przekroju
        z_min = min(np.nanmin(Z_ref), np.nanmin(Z_adj))
        z_max = max(np.nanmax(Z_ref), np.nanmax(Z_adj))
        z_min -= np.abs(0.1 * z_min)
        z_max += np.abs(0.1 * z_max)
        self.cross_plane_item = self.add_cross_section_plane(line_points, z_min, z_max)

        # Dodaj linię profilu
        if line_points is not None and len(line_points) > 1:
            pts = np.array(line_points)
            ref_prof = reference_grid[pts[:,1], pts[:,0]]
            self.ref_profile_line_item = gl.GLLinePlotItem(
                pos=np.column_stack((pts[:,0], pts[:,1], ref_prof)),
                color=(0,0.4,0,1), width=2
            )
            self.view.addItem(self.ref_profile_line_item)

            adj_prof = adjusted_grid[pts[:,1], pts[:,0]] + separation
            self.adj_profile_line_item = gl.GLLinePlotItem(
                pos=np.column_stack((pts[:,0], pts[:,1], adj_prof)),
                color=(0,0,1,1), width=2
            )
            self.view.addItem(self.adj_profile_line_item)

        # Połącz checkboxy z metodami
        self.checkbox_ref.stateChanged.connect(self.toggle_surface_ref)
        self.checkbox_adj.stateChanged.connect(self.toggle_surface_adj)
        self.checkbox_line.stateChanged.connect(self.toggle_profile_line)
        self.checkbox_plane.stateChanged.connect(self.toggle_cross_plane)

        # Wyznacz bounding box (centrowanie widoku)
        all_x = []
        all_y = []
        all_z = []
        all_x.extend(xs)
        all_y.extend(ys)
        all_z.append(np.nanmin(Z_ref))
        all_z.append(np.nanmax(Z_ref))
        if not np.all(np.isnan(Z_adj)):
            all_x.extend(xs)
            all_y.extend(ys)
            all_z.append(np.nanmin(Z_adj))
            all_z.append(np.nanmax(Z_adj))
        if line_points is not None and len(line_points) > 1:
            pts = np.array(line_points)
            all_x.extend(pts[:,0])
            all_y.extend(pts[:,1])
        xc = (min(all_x) + max(all_x)) / 2
        yc = (min(all_y) + max(all_y)) / 2
        zc = (min(all_z) + max(all_z)) / 2 if len(all_z) > 0 else 0
        self.view.setCameraPosition(pos=QtGui.QVector3D(xc, yc, zc))

    def make_voxel_mesh(self, Z, pixel_size=1.0):
        """Tworzy schodkową siatkę voxelową z Z."""
        rows, cols = Z.shape
        verts = []
        faces = []
        colors = []

        def add_quad(v0, v1, v2, v3, color):
            idx = len(verts)
            verts.extend([v0, v1, v2, v3])
            faces.append([idx, idx+1, idx+2])
            faces.append([idx, idx+2, idx+3])
            colors.extend([color]*2)

        for i in range(rows):
            for j in range(cols):
                z = Z[i, j]
                if np.isnan(z):
                    continue
                x0, x1 = j * pixel_size, (j+1) * pixel_size
                y0, y1 = i * pixel_size, (i+1) * pixel_size
                # Kwadrat dla piksela
                v0 = [x0, y0, z]
                v1 = [x1, y0, z]
                v2 = [x1, y1, z]
                v3 = [x0, y1, z]
                add_quad(v0, v1, v2, v3, [0.7, 0.7, 0.7, 1])

                # Ścianki dla różnicy Z z sąsiadem w prawo
                if j < cols-1 and not np.isnan(Z[i, j+1]) and Z[i, j+1] != z:
                    z2 = Z[i, j+1]
                    add_quad(
                        [x1, y0, z], [x1, y0, z2], [x1, y1, z2], [x1, y1, z],
                        [0.5, 0.5, 1, 1]
                    )
                # Ścianki dla różnicy Z z sąsiadem w dół
                if i < rows-1 and not np.isnan(Z[i+1, j]) and Z[i+1, j] != z:
                    z2 = Z[i+1, j]
                    add_quad(
                        [x0, y1, z], [x1, y1, z], [x1, y1, z2], [x0, y1, z2],
                        [0.5, 0.5, 1, 1]
                    )

        verts = np.array(verts)
        faces = np.array(faces)
        colors = np.array(colors)
        mesh = gl.GLMeshItem(vertexes=verts, faces=faces, faceColors=colors, smooth=False, drawEdges=False)
        return mesh

    def add_cross_section_plane(self, line_points, z_min, z_max):
        if line_points is None or len(line_points) < 2:
            return None
        p0 = np.array(line_points[0])
        p1 = np.array(line_points[-1])
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
