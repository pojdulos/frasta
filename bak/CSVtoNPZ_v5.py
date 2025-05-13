import pandas as pd
import numpy as np


from scipy.ndimage import generic_filter
import scipy.ndimage

def fast_local_interpolation(grid, max_iter=10):
    mask = np.isnan(grid)
    for _ in range(max_iter):
        # policz sumę sąsiadów (3x3) ignorując NaN (treat NaN as 0)
        values_sum = scipy.ndimage.convolve(
            np.nan_to_num(grid, nan=0.0),
            weights=np.array([[1,1,1],[1,0,1],[1,1,1]]),
            mode='constant', cval=0.0
        )
        
        # policz liczbę sąsiadów nie-NaN
        valid_neighbors = scipy.ndimage.convolve(
            ~np.isnan(grid),
            weights=np.array([[1,1,1],[1,0,1],[1,1,1]]),
            mode='constant', cval=0.0
        )

        # policz średnią tylko tam, gdzie punkt jest NaN i ma co najmniej 3 sąsiadów
        new_values = values_sum / np.maximum(valid_neighbors, 1)
        to_fill = mask & (valid_neighbors >= 3)
        if not np.any(to_fill):
            break  # koniec jeśli nie ma już punktów do wypełnienia
        grid[to_fill] = new_values[to_fill]
        mask = np.isnan(grid)
    return grid

def local_mean(values):
    center = values[len(values) // 2]
    if np.isnan(center):
        neighbors = values.copy()
        neighbors = neighbors[~np.isnan(neighbors)]
        if len(neighbors) >= 3:  # interpoluj tylko jeśli mamy co najmniej 3 sąsiadów
            return np.mean(neighbors)
    return center



src = "source_data/40b.dat.csv"
dst = "source_data/scan2.npz"

# Wczytanie danych
df = pd.read_csv(src, sep=';', header=None, names=['x', 'y', 'z'])

# Zaokrąglenie w celu stabilności numerycznej
df['x'] = df['x'].round(5)
df['y'] = df['y'].round(5)

# Grupowanie po (x, y) i uśrednianie z
df = df.groupby(['x', 'y'], as_index=False)['z'].mean()

# Sprawdzenie duplikatów po zaokrągleniu (powinno być 0)
dupes = df.duplicated(subset=['x', 'y'])
print("Duplikaty (x, y) po uśrednieniu:", dupes.sum())

from scipy.interpolate import griddata

# 1. Dane wejściowe
points = df[['x', 'y']].values
values = df['z'].values

# 2. Zdefiniuj regularną siatkę (np. 1000 x 1000)
xi = np.linspace(df['x'].min(), df['x'].max(), 1000)
yi = np.linspace(df['y'].min(), df['y'].max(), 1000)
grid_x, grid_y = np.meshgrid(xi, yi)

# 3. Interpolacja
grid_z = griddata(points, values, (grid_x, grid_y), method='linear')  # możesz też użyć 'nearest' lub 'cubic'

# 4. Zapisz do pliku
np.savez(dst, grid=grid_z, x=xi, y=yi)