import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt


def calc_diff_img(scan1, xi, yi, scan2, xi2, yi2, nx, ny):
    x_vals = np.linspace(xi.min(), xi.max(), nx)
    y_vals = np.linspace(yi.min(), yi.max(), ny)

    x2, y2 = np.meshgrid(xi2, yi2)

    points_xyz = np.stack([x2.ravel(), y2.ravel(), scan2.ravel()], axis=1)
    points_xyz = points_xyz[np.isfinite(points_xyz).all(axis=1)]

    xy = points_xyz[:, :2]
    z = points_xyz[:, 2]
    tree = cKDTree(xy)


    # Siatka pomiarowa
    gx, gy = np.meshgrid(x_vals, y_vals)

    # ----> Dla interpolatora (Y, X)
    query_xy_interp = np.stack([gy.ravel(), gx.ravel()], axis=1)

    # ----> Dla KDTree (X, Y)
    query_xy_tree = np.stack([gx.ravel(), gy.ravel()], axis=1)

    # KDTree zapytanie
    dist, idx = tree.query(query_xy_tree, distance_upper_bound=2.0)

    # Odpowiedzi z scan2
    z2 = np.full(query_xy_tree.shape[0], np.nan)
    valid = idx < len(z)
    z2[valid] = z[idx[valid]]

    # Interpolacja scan1
    interp_func = RegularGridInterpolator((yi, xi), scan1, bounds_error=False, fill_value=np.nan)
    z1 = interp_func(query_xy_interp)

    # Różnica tylko tam, gdzie obie wartości są dostępne
    diff = np.full_like(z1, np.nan)
    mask = ~np.isnan(z1) & ~np.isnan(z2)
    diff[mask] = z1[mask] - z2[mask]
    diff_img = diff.reshape((ny, nx))
    return diff_img, x_vals, y_vals



# Wczytaj dane z plików .npz
scan1_data = np.load("source_data/scan1_interp.npz")
scan1 = scan1_data['grid']

# Parametry siatki (upewnij się, że są zapisane w plikach)
pixel_size_x = scan1_data.get('px_x', 0.00138)
pixel_size_y = scan1_data.get('px_y', 0.00138)
xi = scan1_data['xi']  # długość = szerokość siatki (X)
yi = scan1_data['yi']  # długość = wysokość siatki (Y)

print("scan1 shape:", scan1.shape)
print("len xi:", len(xi), "| len yi:", len(yi))

scan2_data = np.load("source_data/scan2_interp.npz")
scan2 = scan2_data['grid']
scan2 = np.flipud(scan2)
scan2 = -scan2

xi2 = scan2_data['xi']  # oś X dla scan2
yi2 = scan2_data['yi']  # oś Y dla scan2
yi2 = yi2[::-1]

print("scan2 shape:", scan2.shape)
print("len xi2:", len(xi2))
print("len yi2:", len(yi2))


plt.imshow(scan2, cmap='seismic', origin='lower',vmin=-1500, vmax=+1500)
plt.imshow(scan1, cmap='gray', origin='lower',vmin=-1500, vmax=+1500)
plt.colorbar(label='Z')
plt.title("scan2")
plt.show()

# Siatka pomiarowa 400x300 w obszarze scan1
nx, ny = 400, 300

diff_img, x_vals, y_vals = calc_diff_img(scan1, xi, yi, scan2, xi2, yi2, nx, ny)

# Wizualizacja (opcjonalna)
plt.imshow(diff_img, cmap='seismic', #origin='lower',
           extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()],
           vmin=-1500, vmax=+1500)
plt.colorbar(label='Z difference')
plt.title("Różnica Z (KDTree)")
plt.show()
