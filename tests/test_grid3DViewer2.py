import pytest
import numpy as np
from PyQt5 import QtWidgets
from src.grid3DViewer2 import Grid3DViewer

@pytest.fixture
def viewer():
    return Grid3DViewer()

def test_init_controls_creates_controls(viewer):
    assert hasattr(viewer, 'controls_panel')
    assert hasattr(viewer, 'checkbox_ref')
    assert hasattr(viewer, 'checkbox_adj')
    assert hasattr(viewer, 'checkbox_line')
    assert hasattr(viewer, 'checkbox_plane')

def test_compute_auto_lo_hi_returns_percentiles(viewer):
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    lo, hi = viewer._compute_auto_lo_hi(arr)
    assert lo < hi
    assert isinstance(lo, float)
    assert isinstance(hi, float)

def test_update_range_widgets_ref(viewer):
    viewer._update_range_widgets('ref', 1.0, 2.0, auto=True)
    assert viewer.spin_lo_ref.value() == 1.0
    assert viewer.spin_hi_ref.value() == 2.0
    assert viewer.chk_auto_ref.isChecked()
    assert not viewer.spin_lo_ref.isEnabled()
    assert not viewer.spin_hi_ref.isEnabled()

def test_update_range_widgets_adj(viewer):
    viewer._update_range_widgets('adj', 3.0, 4.0, auto=False)
    assert viewer.spin_lo_adj.value() == 3.0
    assert viewer.spin_hi_adj.value() == 4.0
    assert not viewer.chk_auto_adj.isChecked()
    assert viewer.spin_lo_adj.isEnabled()
    assert viewer.spin_hi_adj.isEnabled()

def test_get_lo_hi_for_auto(viewer):
    arr = np.array([[1, 2], [3, 4]])
    viewer.range_ref_auto = True
    lo, hi = viewer._get_lo_hi_for('ref', arr)
    assert lo < hi

def test_get_lo_hi_for_manual(viewer):
    arr = np.array([[1, 2], [3, 4]])
    viewer.range_ref_auto = False
    viewer.range_ref = (1.5, 3.5)
    lo, hi = viewer._get_lo_hi_for('ref', arr)
    assert lo == 1.5
    assert hi == 3.5

def test_prepare_reference_surface_downsamples_and_masks(viewer):
    arr = np.random.rand(20, 20)
    arr[0, 0] = np.nan
    arr[1, 1] = 1e7
    xs, ys, Z, xs_idx, ys_idx = viewer._prepare_reference_surface(arr, max_points=10)
    assert isinstance(xs, np.ndarray)
    assert isinstance(ys, np.ndarray)
    assert isinstance(Z, np.ndarray)
    assert np.isnan(Z[0, 0]) or np.isnan(Z[1, 1])

def test_remove_existing_items_resets_items(viewer):
    viewer.surface_ref_item = object()
    viewer.surface_adj_item = object()
    viewer.ref_profile_line_item = object()
    viewer.adj_profile_line_item = object()
    viewer.cross_plane_item = object()
    viewer._lod['ref'] = None
    viewer._lod['adj'] = None
    viewer.remove_existing_items()
    assert viewer.surface_ref_item is None
    assert viewer.surface_adj_item is None
    assert viewer.ref_profile_line_item is None
    assert viewer.adj_profile_line_item is None
    assert viewer.cross_plane_item is None

# def test_update_data_handles_empty_grids(viewer):
#     arr = np.full((10, 10), np.nan)
#     viewer.update_data(arr)
#     assert viewer._ref_last is not None

def test_update_data_with_adjusted_grid(viewer):
    arr = np.random.rand(10, 10)
    adj = np.random.rand(10, 10)
    viewer.update_data(arr, adj)
    assert viewer._ref_last is not None
    assert viewer._adj_last is not None