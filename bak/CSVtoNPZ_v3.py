import pandas as pd
import numpy as np


from scipy.ndimage import generic_filter

def local_mean(values):
    center = values[len(values) // 2]
    if np.isnan(center):
        neighbors = values.copy()
        neighbors = neighbors[~np.isnan(neighbors)]
        if len(neighbors) >= 3:  # interpoluj tylko jeśli mamy co najmniej 3 sąsiadów
            return np.mean(neighbors)
    return center



src = "source_data/40a.dat.csv"
dst = "source_data/scan1.npz"

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

# Unikalne współrzędne
x_unique, x_inv = np.unique(df['x'].values, return_inverse=True)
y_unique, y_inv = np.unique(df['y'].values, return_inverse=True)

# Tworzenie siatki
grid = np.full((len(y_unique), len(x_unique)), np.nan, dtype=np.float32)
grid[y_inv, x_inv] = df['z'].values

print("Wszystkich punktów:", len(df))
print("Wartości nie-NaN w siatce:", np.count_nonzero(~np.isnan(grid)))

# Wykonujemy lokalną interpolację 3x3, wielokrotnie aż do konwergencji
for _ in range(5):  # liczba iteracji może być większa jeśli potrzebujesz dokładniejszego wypełnienia
    grid = generic_filter(grid, local_mean, size=3, mode='constant', cval=np.nan)

print("Wszystkich punktów:", len(df))

# Zapis do pliku
np.savez(dst, grid=grid, x_unique=x_unique, y_unique=y_unique)
