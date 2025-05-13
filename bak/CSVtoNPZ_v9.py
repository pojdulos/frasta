import pandas as pd
import numpy as np

from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import numpy as np

src = "source_data/40a.dat.csv"
dst = "source_data/scan1.npz"

# Załaduj dane
df = pd.read_csv(src, sep=';', header=None, names=['x', 'y', 'z'])

print(len(df))

data = df.values #.astype(np.float32)

# Wyciąganie współrzędnych
x, y, z = data[:, 0], data[:, 1], data[:, 2]

# Obliczanie rozmiaru piksela
dx = np.diff(np.sort(np.unique(x)))
dy = np.diff(np.sort(np.unique(y)))

pixel_size_x = np.median(dx[dx > 0])
pixel_size_y = np.median(dy[dy > 0])

print(f"Rozmiar piksela: {pixel_size_x} x {pixel_size_y}")

pixel_size_x = 1.38
pixel_size_y = 1.38

print(f"Rozmiar piksela: {pixel_size_x} x {pixel_size_y}")


# Wymiary siatki
x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()

print(f"x_min: {x_min}, y_min: {y_min}")
print(f"x_max: {x_max}, y_min: {y_max}")


grid_size_x = int((x_max - x_min) / pixel_size_x) + 1
grid_size_y = int((y_max - y_min) / pixel_size_y) + 1

print(f"Rozmiar siatki: {grid_size_x} x {grid_size_y}")

# Tworzenie siatki regularnej
grid_y, grid_x = np.mgrid[
    y_min:y_max:complex(grid_size_y),
    x_min:x_max:complex(grid_size_x)
]

# Szybka interpolacja
grid_z = griddata((x, y), z, (grid_x, grid_y), method='nearest')
#grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')

# Filtruj punkty daleko od danych
# print("Filtrowanie po odległości...")
# tree = cKDTree(np.column_stack((x, y)))
# points_flat = np.column_stack((grid_x.ravel(), grid_y.ravel()))
# dists, _ = tree.query(points_flat)

# # Próg: np. 3 x pixel_size (w jednostkach świata)
# dist_threshold = 50.0 * max(pixel_size_x, pixel_size_y)
# mask = dists.reshape(grid_z.shape) > dist_threshold
# grid_z[mask] = np.nan

# Zapis
np.savez(dst, grid=grid_z)

