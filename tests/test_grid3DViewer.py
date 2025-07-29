import numpy as np
from src.grid3DViewer import Grid3DViewer

def test_prepare_reference_surface():
    viewer = Grid3DViewer()
    grid = np.arange(100).reshape(10, 10).astype(float)
    xs, ys, Z_ref = viewer._prepare_reference_surface(grid)
    assert Z_ref.shape[0] == len(ys)
    assert Z_ref.shape[1] == len(xs)
