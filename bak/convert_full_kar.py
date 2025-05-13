# === Wczytywanie danych ===
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, shift
import matplotlib.patches as patches
import os
import pandas as pd

# Wczytanie danych
df_a = pd.read_csv('source_data/40a.dat.csv', sep=';', header=None, quotechar='"')
df_b = pd.read_csv('source_data/40b.dat.csv', sep=';', header=None, quotechar='"')

L = np.array(df_a.values.tolist(), dtype=np.float32)
S = np.array(df_a.values.tolist(), dtype=np.float32)

# Obliczanie rozmiaru piksela z danych L
L_x_sorted = np.sort(L[:, 0])
L_y_sorted = np.sort(L[:, 1])
L_dx = np.diff(L_x_sorted)
L_dy = np.diff(L_y_sorted)
pixel_size_x_L = np.median(L_dx[L_dx > 0])
pixel_size_y_L = np.median(L_dy[L_dy > 0])

# Obliczanie rozmiaru piksela z danych S
S_x_sorted = np.sort(S[:, 0])
S_y_sorted = np.sort(S[:, 1])
S_dx = np.diff(S_x_sorted)
S_dy = np.diff(S_y_sorted)
pixel_size_x_S = np.median(S_dx[S_dx > 0])
pixel_size_y_S = np.median(S_dy[S_dy > 0])

# Średni rozmiar piksela
pixel_size_x = np.mean([pixel_size_x_L, pixel_size_x_S])
pixel_size_y = np.mean([pixel_size_y_L, pixel_size_y_S])

print(f"Rozmiar piksela X: {pixel_size_x:.8f} um")
print(f"Rozmiar piksela Y: {pixel_size_y:.8f} um")

# Przesunięcia w mm (obliczone na podstawie rozmiaru piksela i przesunięć w pikselach)
# pixel_size_mm = 0.00276
shift_L_mm = (40 * pixel_size_x, 35 * pixel_size_y)
shift_S_mm = (15 * pixel_size_x, 15 * pixel_size_y)

# Przesuwanie danych źródłowych
L[:, 0] += shift_L_mm[0]  # X
L[:, 1] += shift_L_mm[1]  # Y

S[:, 0] += shift_S_mm[0]  # X
S[:, 1] += shift_S_mm[1]  # Y


# Ustalenie wspólnego obszaru siatki
x_min, x_max = min(L[:,0].min(), S[:,0].min()), max(L[:,0].max(), S[:,0].max())
y_min, y_max = min(L[:,1].min(), S[:,1].min()), max(L[:,1].max(), S[:,1].max())

# Liczba punktów siatki na podstawie rozmiaru piksela
grid_size_x = int((x_max - x_min) / pixel_size_x) + 1
grid_size_y = int((y_max - y_min) / pixel_size_y) + 1

print(f"Rozmiar siatki: {grid_size_x} x {grid_size_y} pikseli")

# Tworzenie siatki
grid_x, grid_y = np.mgrid[
    x_min:x_max:complex(grid_size_x),
    y_min:y_max:complex(grid_size_y)
]

# Interpolacja
L_grid = griddata((L[:,0], L[:,1]), L[:,2], (grid_x, grid_y), method='nearest')
S_grid = griddata((S[:,0], S[:,1]), S[:,2], (grid_x, grid_y), method='nearest')

vmin = max(np.nanmin(L_grid), np.nanmin(S_grid))
vmax = min(np.nanmax(L_grid), np.nanmax(S_grid))

print(f"vmin={vmin}, vmax={vmax}")

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

im1 = axs[0].imshow(L_grid, cmap='gray', vmin=vmin, vmax=vmax)
axs[0].set_title('L_grid_transformed')
plt.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)

im2 = axs[1].imshow(S_grid, cmap='gray', vmin=vmin, vmax=vmax)
axs[1].set_title('S_grid_transformed')
plt.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)

plt.show()


# # Zapis siatek
# np.save("L_big_cloud.npy", L_grid_transformed)
# np.save("S_big_cloud.npy", S_grid_transformed)

# # Zapis metadanych
# metadata = {
#     'pixel_size_x_mm': pixel_size_x,
#     'pixel_size_y_mm': pixel_size_y,
#     'x_min': x_min,
#     'x_max': x_max,
#     'y_min': y_min,
#     'y_max': y_max,
#     'grid_size_x': grid_size_x,
#     'grid_size_y': grid_size_y
# }
# np.save("grid_metadata.npy", metadata)
