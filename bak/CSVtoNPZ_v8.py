import pandas as pd
import numpy as np

from scipy.interpolate import griddata
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree
from matplotlib.path import Path

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.spatial import cKDTree, ConvexHull
from matplotlib.path import Path

def interpolate_grid(x, y, z, pixel_size_x=None, pixel_size_y=None, scale=1.0, max_nearest_distance=None):
    """
    Tworzy interpolowaną siatkę 2D z danych (x, y, z).
    - Interpolacja 'linear' wewnątrz obiektu.
    - 'nearest' tylko dla bliskich punktów (fallback dla NaN).
    - Maskowanie punktów poza convex hull.
    
    Parametry:
        x, y, z : 1D arrays – dane wejściowe
        pixel_size_x, pixel_size_y : float lub None – opcjonalnie wymuszony krok siatki
        scale : float – skalowanie rozdzielczości siatki (np. 0.5 = o połowę mniejsza)
        max_nearest_distance : float lub None – maks. odległość do zastosowania fallback 'nearest'
    
    Zwraca:
        grid_z : ndarray – interpolowana siatka
        xi, yi : 1D arrays – współrzędne siatki
    """
    # Krok siatki
    if pixel_size_x is None:
        dx = np.diff(np.sort(np.unique(x)))
        pixel_size_x = np.median(dx[dx > 0])
    if pixel_size_y is None:
        dy = np.diff(np.sort(np.unique(y)))
        pixel_size_y = np.median(dy[dy > 0])
        
    pixel_size_x /= scale
    pixel_size_y /= scale

    # Wymiary siatki
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    grid_size_x = int((x_max - x_min) / pixel_size_x) + 1
    grid_size_y = int((y_max - y_min) / pixel_size_y) + 1

    xi = np.linspace(x_min, x_max, grid_size_x)
    yi = np.linspace(y_min, y_max, grid_size_y)
    grid_x, grid_y = np.meshgrid(xi, yi)

    print(f"Siatka: {grid_size_x} x {grid_size_y} punktów")

    # Interpolacja liniowa
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')

    # Fallback 'nearest' tylko dla bliskich NaN
    if max_nearest_distance is None:
        max_nearest_distance = 3 * max(pixel_size_x, pixel_size_y)

    mask_nan = np.isnan(grid_z)
    if np.any(mask_nan):
        print("Uzupełnianie NaN metodą 'nearest' dla sąsiednich punktów...")
        tree = cKDTree(np.column_stack((x, y)))
        target_points = np.column_stack((grid_x[mask_nan], grid_y[mask_nan]))
        dists, idxs = tree.query(target_points, distance_upper_bound=max_nearest_distance)

        nearest_values = np.full_like(target_points[:, 0], np.nan, dtype=np.float32)
        valid = ~np.isinf(dists)
        nearest_values[valid] = z[idxs[valid]]

        grid_z[mask_nan] = nearest_values

    # Odcinanie wartości poza convex hull
    print("Maskowanie punktów poza obiektem...")
    hull = ConvexHull(np.column_stack((x, y)))
    polygon = Path(np.column_stack((x, y))[hull.vertices])
    points_flat = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    inside = polygon.contains_points(points_flat).reshape(grid_z.shape)
    grid_z[~inside] = np.nan

    return grid_z, xi, yi


src = "source_data/40a.dat.csv"
dst = "source_data/scan1.npz"

# Załaduj dane
df = pd.read_csv(src, sep=';', header=None, names=['x', 'y', 'z'])

df = df.dropna(subset=['z'])  # ważne: usuwamy NaN-y z danych

x = df['x'].values
y = df['y'].values
z = df['z'].values

# Interpolacja
grid_z, xi, yi = interpolate_grid(x, y, z, scale=0.5)

# Zapis
np.savez(dst, grid=grid_z)

