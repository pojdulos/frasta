import numpy as np
import pytest
from src.grid3DViewer import Grid3DViewer

@pytest.fixture
def viewer():
    return Grid3DViewer()

def test_prepare_reference_surface_downsampling_and_nan(viewer):
    grid = np.arange(100).reshape(10, 10).astype(float)
    grid[0, 0] = np.nan
    xs, ys, Z_ref = viewer._prepare_reference_surface(grid)
    assert Z_ref.shape[0] == len(ys)
    assert Z_ref.shape[1] == len(xs)
    assert np.isnan(Z_ref[0, 0])

def test_prepare_adjusted_surface_with_none(viewer):
    grid = np.ones((10, 10))
    xs = ys = np.arange(0, 10, 2)
    Z_ref = np.ones((5, 5))
    Z_adj = viewer._prepare_adjusted_surface(None, ys, xs, 0, Z_ref)
    assert np.all(np.isnan(Z_adj))

def test_prepare_adjusted_surface_with_data(viewer):
    grid = np.ones((10, 10))
    xs = ys = np.arange(0, 10, 2)
    Z_ref = np.ones((5, 5))
    Z_adj = viewer._prepare_adjusted_surface(grid, ys, xs, 2, Z_ref)
    assert np.allclose(Z_adj, 3)

def test_compute_z_limits_with_adjusted(viewer):
    Z_ref = np.array([[1, 2], [3, 4]])
    Z_adj = np.array([[5, 6], [7, 8]])
    z_min, z_max = viewer._compute_z_limits(Z_ref, Z_adj, True)
    assert z_min == 1
    assert z_max == 8

def test_compute_z_limits_without_adjusted(viewer):
    Z_ref = np.array([[1, 2], [3, 4]])
    Z_adj = np.array([[np.nan, np.nan], [np.nan, np.nan]])
    z_min, z_max = viewer._compute_z_limits(Z_ref, Z_adj, False)
    assert z_min == 1
    assert z_max == 4

def test_create_verts_grid_shape(viewer):
    Z = np.ones((3, 3))
    xs = np.arange(3)
    ys = np.arange(3)
    verts_grid = viewer.create_verts_grid(Z, xs, ys, 3, 3)
    assert verts_grid.shape == (4, 4, 3)

def test_make_voxel_mesh_returns_mesh(viewer):
    Z = np.ones((3, 3))
    xs = np.arange(3)
    ys = np.arange(3)
    mesh = viewer.make_voxel_mesh(Z, xs, ys)
    assert mesh is not None

def test_make_voxel_mesh_returns_none_for_all_nan(viewer):
    Z = np.full((3, 3), np.nan)
    xs = np.arange(3)
    ys = np.arange(3)
    mesh = viewer.make_voxel_mesh(Z, xs, ys)
    assert mesh is None
