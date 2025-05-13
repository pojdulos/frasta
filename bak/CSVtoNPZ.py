import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, shift
import matplotlib.patches as patches
import os
import pandas as pd
import cv2


src = "source_data/40a.dat.csv"
dst = "source_data/scan1.npz"

df = pd.read_csv(src, sep=';', header=None, names=['x', 'y', 'z'])

dupes = df.duplicated(subset=['x', 'y'])
print("Duplikaty (x, y):", dupes.sum())


# 1. Unikalne wartości i indeksy
x_unique, x_inv = np.unique(df['x'].values, return_inverse=True)
y_unique, y_inv = np.unique(df['y'].values, return_inverse=True)

# 2. Siatka Z
grid = np.full((len(y_unique), len(x_unique)), np.nan, dtype=np.float32)


grid[y_inv, x_inv] = df['z'].values

print("Wszystkich punktów:", len(df))
print("Wartości nie-NaN w siatce:", np.count_nonzero(~np.isnan(grid)))

# 3. Zapis do pliku .npz
np.savez(dst, grid=grid, x_unique=x_unique, y_unique=y_unique)

# plt.imshow(grid, cmap='gray', origin='lower')
# plt.colorbar()
# plt.title('Scan 1')
# plt.show()    

