import numpy as np
import pandas as pd
import h5py
from scipy.spatial import cKDTree
import os
import matplotlib.pyplot as plt

def load_and_display(filepath):
    """
    Funkcja wczytuje siatki L_grid i S_grid z pliku HDF5 i je wyświetla.
    """
    with h5py.File(filepath, 'r') as f:
        L_grid = f['L_grid'][:]
        S_grid = f['S_grid'][:]

        # Wczytaj zakres wartości do wyrównania skalowania obrazów
        vmin = min(np.nanmin(L_grid), np.nanmin(S_grid))
        vmax = max(np.nanmax(L_grid), np.nanmax(S_grid))

    S_grid = np.fliplr(S_grid)
    S_grid = -S_grid

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    im1 = axs[0].imshow(L_grid, cmap='gray', vmin=vmin, vmax=vmax)
    axs[0].set_title('L_grid')
    plt.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)

    im2 = axs[1].imshow(S_grid, cmap='gray', vmin=vmin, vmax=vmax)
    axs[1].set_title('S_grid')
    plt.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

import h5py
import matplotlib.pyplot as plt
import numpy as np

def load_and_display_with_diff(filepath):
    """
    Funkcja wczytuje siatki L_grid i S_grid z pliku HDF5,
    wyświetla je oraz obraz różnicy.
    """
    with h5py.File(filepath, 'r') as f:
        L_grid = f['L_grid'][:]
        S_grid = f['S_grid'][:]

    S_grid = np.fliplr(S_grid)
    S_grid = -S_grid

    diff_grid = L_grid - S_grid

    # Ustal skalę wspólną dla L_grid i S_grid
    vmin = min(np.nanmin(L_grid), np.nanmin(S_grid))
    vmax = max(np.nanmax(L_grid), np.nanmax(S_grid))

    # Skala dla różnicy (oddzielna, symetryczna względem zera)
    diff_abs_max = np.max(np.abs(diff_grid))

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # L_grid
    im1 = axs[0].imshow(L_grid, cmap='gray', vmin=vmin, vmax=vmax)
    axs[0].set_title('L_grid')
    plt.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)

    # S_grid
    im2 = axs[1].imshow(S_grid, cmap='gray', vmin=vmin, vmax=vmax)
    axs[1].set_title('S_grid')
    plt.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)

    # Różnica
    im3 = axs[2].imshow(diff_grid, cmap='bwr', vmin=-diff_abs_max, vmax=diff_abs_max)
    axs[2].set_title('L_grid - S_grid (różnica)')
    plt.colorbar(im3, ax=axs[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

# Przykładowe użycie:
# load_and_display_with_diff('output/full_data.h5')


# === Wczytywanie danych ===
def load_and_convert_to_hdf5():
    # Wczytaj pliki
    print("Wczytywanie danych...")
    df_a = pd.read_csv('source_data/40a.dat.csv', sep=';', header=None, quotechar='"')
    df_b = pd.read_csv('source_data/40b.dat.csv', sep=';', header=None, quotechar='"')

    # Na numpy array
    L = np.array(df_a.values.tolist(), dtype=np.float32)
    S = np.array(df_b.values.tolist(), dtype=np.float32)

    # Obliczanie rozmiaru piksela
    print("Obliczanie rozmiaru piksela...")
    L_dx = np.diff(np.sort(L[:, 0]))
    L_dy = np.diff(np.sort(L[:, 1]))
    S_dx = np.diff(np.sort(S[:, 0]))
    S_dy = np.diff(np.sort(S[:, 1]))

    pixel_size_x = np.mean([
        np.median(L_dx[L_dx > 0]),
        np.median(S_dx[S_dx > 0])
    ])
    pixel_size_y = np.mean([
        np.median(L_dy[L_dy > 0]),
        np.median(S_dy[S_dy > 0])
    ])

    print(f"Rozmiar piksela X: {pixel_size_x:.8f} um")
    print(f"Rozmiar piksela Y: {pixel_size_y:.8f} um")

    # Przesunięcia
    print("Przesuwanie danych...")
    shift_L = (40 * pixel_size_x, 35 * pixel_size_y)
    shift_S = (15 * pixel_size_x, 15 * pixel_size_y)

    L[:, 0] += shift_L[0]
    L[:, 1] += shift_L[1]

    S[:, 0] += shift_S[0]
    S[:, 1] += shift_S[1]

    # Ustalenie wspólnego obszaru siatki
    x_min, x_max = min(L[:, 0].min(), S[:, 0].min()), max(L[:, 0].max(), S[:, 0].max())
    y_min, y_max = min(L[:, 1].min(), S[:, 1].min()), max(L[:, 1].max(), S[:, 1].max())

    # Rozmiar siatki
    grid_size_x = int((x_max - x_min) / pixel_size_x) + 1
    grid_size_y = int((y_max - y_min) / pixel_size_y) + 1

    print(f"Rozmiar siatki: {grid_size_x} x {grid_size_y}")

    # Tworzenie siatki współrzędnych
    grid_x = np.linspace(x_min, x_max, grid_size_x, dtype=np.float32)
    grid_y = np.linspace(y_min, y_max, grid_size_y, dtype=np.float32)
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y, indexing='ij')

    # Przygotowanie punktów siatki do interpolacji
    grid_points = np.vstack([mesh_x.ravel(), mesh_y.ravel()]).T

    # Interpolacja za pomocą KDTree
    print("Interpolacja L...")
    L_tree = cKDTree(L[:, :2])
    _, L_idx = L_tree.query(grid_points)
    L_grid = L[L_idx, 2].reshape((grid_size_x, grid_size_y))

    print("Interpolacja S...")
    S_tree = cKDTree(S[:, :2])
    _, S_idx = S_tree.query(grid_points)
    S_grid = S[S_idx, 2].reshape((grid_size_x, grid_size_y))

    # Zapis do HDF5
    print("Zapisywanie do HDF5...")
    os.makedirs('output', exist_ok=True)
    with h5py.File('output/full_data.h5', 'w') as f:
        f.create_dataset('L_grid', data=L_grid, compression='gzip')
        f.create_dataset('S_grid', data=S_grid, compression='gzip')
        # Metadane jako atrybuty
        f.attrs['pixel_size_x'] = pixel_size_x
        f.attrs['pixel_size_y'] = pixel_size_y
        f.attrs['x_min'] = x_min
        f.attrs['x_max'] = x_max
        f.attrs['y_min'] = y_min
        f.attrs['y_max'] = y_max
        f.attrs['grid_size_x'] = grid_size_x
        f.attrs['grid_size_y'] = grid_size_y

    print("Gotowe!")

# === Przykład odczytu danych i metadanych ===
# with h5py.File('output/full_data.h5', 'r') as f:
#     L_grid = f['L_grid'][:]
#     S_grid = f['S_grid'][:]
#     pixel_size_x = f.attrs['pixel_size_x']
#     grid_size_x = f.attrs['grid_size_x']
#     np.save('L_grid.npy', L_grid)  # np. zapis fragmentu na szybko


load_and_display_with_diff('output/full_data.h5')

