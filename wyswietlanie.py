import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Wczytaj dane
df = pd.read_csv('source_data/3-0b x5 conf.dat', sep=';', header=None, names=['x','y','z'], quotechar='"')

x, y, z = df['x'].values, df['y'].values, df['z'].values

dx = np.diff(np.sort(np.unique(x)))
dy = np.diff(np.sort(np.unique(y)))
px_x = np.median(dx[dx > 0]).round(2)
px_y = np.median(dy[dy > 0]).round(2)

print(f"px: {px_x}, {px_y}")

x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()
grid_size_x = int((x_max - x_min) / px_x) + 1
grid_size_y = int((y_max - y_min) / px_y) + 1

grid = np.full((grid_size_y, grid_size_x), np.nan, dtype=np.float32)

print(f"grid shape: {grid.shape}")

# Zaokrąglamy wartości X i Y do pozycji na siatce
for xi, yi, zi in zip(x, y, z):
    idx = int(round((xi - x_min) / px_x))
    idy = int(round((yi - y_min) / px_y))
    grid[idy, idx] = zi

xi_grid = np.linspace(x_min, x_max, grid_size_x)
yi_grid = np.linspace(y_min, y_max, grid_size_y)

print(f"xi_grid: {xi_grid}, yi_grid: {yi_grid}")

X, Y = np.meshgrid(xi_grid, yi_grid)

to_save = dict(grid=grid, xi=xi_grid, yi=yi_grid, px_x=px_x, px_y=px_y, x_data=x, y_data=y, z_data=z)
np.savez('source_data/moj_test_B.npz', **to_save)
