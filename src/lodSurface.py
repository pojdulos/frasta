import numpy as np
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtWidgets, QtGui, QtCore
import pyqtgraph as pg


class LODSurface:
    """Zarządza kilkoma GLMeshItem o różnych gęstościach (krokach) i
    przełącza widoczny poziom LOD w zależności od zoomu/kamery."""
    def __init__(self, view, steps=(1,2,4,8,16), shader=None):
        self.view = view
        self.steps = tuple(sorted(set(steps)))
        self.shader = shader
        self.items = {}   # step -> GLMeshItem
        self.data = None  # (xs, ys, Z)
        self.color = (0,1,0,1)
        self.colormap = 'RG'
        self.lohi = None  # (lo,hi)
        self.mode = 'surface'
        self.visible = True

        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self.update_lod)
        self._timer.start(33)

    def destroy(self):
        for it in self.items.values():
            try: self.view.removeItem(it)
            except Exception: pass
        self.items.clear()
        self._timer.stop()

    def set_visible(self, on: bool):
        self.visible = bool(on)
        for it in self.items.values():
            it.setVisible(self.visible and (it is self._current_item()))

    def set_data(self, xs, ys, Z):
        """Zapycha dane; nie tworzy od razu wszystkich LODów (lazy)."""
        self.data = (xs, ys, Z.astype(np.float32, copy=False))
        # jeśli już coś było, odśwież kolory/tryb tylko dla aktualnego itemu
        self._ensure_current_exists()
        self._restyle_all_existing()

    def update_style(self, mode, colormap, base_color, lo, hi):
        self.mode = mode
        self.colormap = colormap
        self.color = base_color
        self.lohi = (float(lo), float(hi))
        self._restyle_all_existing()

    # ---------- wewnętrzne ----------
    def _ensure_current_exists(self):
        s = self._pick_step()
        if s not in self.items:
            self.items[s] = self._build_item_for_step(s)

    def _current_item(self):
        if not self.items: return None
        # znajdź jedyny widoczny
        for s,it in self.items.items():
            if it.isVisible(): return it
        # albo zwróć ostatni tworzony
        return next(iter(self.items.values()))

    def _restyle_all_existing(self):
        for s,it in self.items.items():
            self._apply_style(it, s)

    def _build_item_for_step(self, step):
        xs, ys, Z = self.data
        if xs is None: xs = np.arange(Z.shape[1], dtype=np.float32)
        if ys is None: ys = np.arange(Z.shape[0], dtype=np.float32)

        X, Y = np.meshgrid(xs[::step], ys[::step], indexing='xy')
        H = Z[::step, ::step]
        h, w = H.shape

        # wierzchołki i trójkąty
        V = np.c_[X.ravel(), Y.ravel(), H.ravel()].astype(np.float32)
        idx = np.arange(h*w, dtype=np.uint32).reshape(h, w)
        f1 = np.c_[idx[:-1,:-1].ravel(), idx[1:,:-1].ravel(), idx[1:,1:].ravel()]
        f2 = np.c_[idx[:-1,:-1].ravel(), idx[1:,1:].ravel(), idx[:-1,1:].ravel()]
        faces = np.vstack([f1, f2])

        md = gl.MeshData(vertexes=V, faces=faces)
        it = gl.GLMeshItem(meshdata=md, smooth=True, drawEdges=False, drawFaces=True, shader='shaded')
        it._mesh = md   # przechowuj MeshData w itemie (kompatybilnie wstecz)

        if self.shader is not None:
            it.setShader(self.shader)

        self.view.addItem(it)
        it.setVisible(False)
        # kolory/tryb będą nałożone w _apply_style
        return it

    def _apply_style(self, it, step):
        # tryb
        if self.mode == 'wireframe':
            it.opts['drawFaces'] = False
            it.opts['drawEdges'] = True
        else:
            it.opts['drawFaces'] = True
            it.opts['drawEdges'] = False

        # kolory
        md = getattr(it, '_mesh', None)
        if md is None:
            # fallback na wszelki wypadek, gdyby item nie był utworzony przez LODSurface
            try:
                md = it.meshData()  # w niektórych wersjach istnieje
            except Exception:
                return  # brak dostępu do MeshData – nic nie zrobimy bez przebudowy

        V = md.vertexes(indexed=False)
        # ... wylicz C ...
        md.setVertexColors(C)

        # W części wersji pyqtgraph samo setVertexColors nie odświeża buforów — wymuś aktualizację:
        it.setMeshData(meshdata=md)
        # lub przynajmniej:
        # it.update()

    def _pick_step(self):
        # prosty heurystyczny wybór kroku na podstawie kamery
        view = self.view
        fov = np.deg2rad(view.opts.get('fov', 60.0))
        dist = float(view.opts['distance'])
        px_h = max(1, view.height())
        px_per_unit = px_h / (2.0*np.tan(fov/2.0)*dist)

        # oszacuj wielkość komórki (przyjmij ~1, bo ważna jest relacja)
        cell_px = px_per_unit * 1.0
        desired = max(1, int(np.ceil(1.5 / max(cell_px, 1e-6))))
        # najbliższy dostępny step
        return min(self.steps, key=lambda s: abs(s - desired))

    def update_lod(self):
        if self.data is None or not self.visible:
            return
        s = self._pick_step()
        if s not in self.items:
            self.items[s] = self._build_item_for_step(s)
            self._apply_style(self.items[s], s)
        # przełącz widoczność
        for k,it in self.items.items():
            it.setVisible(self.visible and (k == s))
