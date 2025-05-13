import pandas as pd
import numpy as np

from scipy.interpolate import griddata
from scipy.interpolate import LinearNDInterpolator

def generate_regular_grid(df, scale=1.0):
    x = df['x'].values
    y = df['y'].values
    z = df['z'].values

    # Oblicz minimalny krok
    x_sorted = np.sort(np.unique(x))
    y_sorted = np.sort(np.unique(y))
    dx = np.diff(x_sorted)
    dy = np.diff(y_sorted)
    min_dx = np.min(dx[dx > 0])  # pomiń zera
    min_dy = np.min(dy[dy > 0])

    # Wymiary siatki
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    nx = int(((x_max - x_min) / min_dx) * scale)
    ny = int(((y_max - y_min) / min_dy) * scale)

    print(f"Rozdzielczość siatki: {nx} x {ny}")

    xi = np.linspace(x_min, x_max, nx)
    yi = np.linspace(y_min, y_max, ny)
    grid_x, grid_y = np.meshgrid(xi, yi)

    # Interpolacja
    #grid_z = griddata(points=(x, y), values=z, xi=(grid_x, grid_y), method='linear')

    #interp = LinearNDInterpolator(list(zip(x, y)), z)
   # grid_z = interp(grid_x, grid_y)

    return grid_z, xi, yi


src = "source_data/40a.dat.csv"
dst = "source_data/scan1.npz"

# Załaduj dane
df = pd.read_csv(src, sep=';', header=None, names=['x', 'y', 'z'])
data = df.values.astype(np.float32)

# Wyciąganie współrzędnych
x, y, z = data[:, 0], data[:, 1], data[:, 2]

# Obliczanie rozmiaru piksela
dx = np.diff(np.sort(np.unique(x)))
dy = np.diff(np.sort(np.unique(y)))
pixel_size_x = np.median(dx[dx > 0])
pixel_size_y = np.median(dy[dy > 0])

# Wymiary siatki
x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()

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

# Zapis
np.savez(dst, grid=grid_z)

