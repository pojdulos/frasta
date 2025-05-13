import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
from matplotlib.path import Path

def build_grid_direct(df, pixel_size_x, pixel_size_y):
    x = df['x'].values
    y = df['y'].values
    z = df['z'].values

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    print(f"x_min: {x_min}, y_min: {y_min}")
    print(f"x_max: {x_max}, y_min: {y_max}")

    grid_size_x = int((x_max - x_min) / pixel_size_x) + 1
    grid_size_y = int((y_max - y_min) / pixel_size_y) + 1

    grid_z = np.full((grid_size_y, grid_size_x), np.nan, dtype=np.float32)
    counts = np.zeros_like(grid_z, dtype=np.int32)

    print(f"teraz pętla...")

    # Przypisywanie punktów
    for xi, yi, zi in zip(x, y, z):
        ix = int(round((xi - x_min) / pixel_size_x))
        iy = int(round((yi - y_min) / pixel_size_y))

        if 0 <= ix < grid_size_x and 0 <= iy < grid_size_y:
            if np.isnan(grid_z[iy, ix]):
                grid_z[iy, ix] = zi
            else:
                grid_z[iy, ix] += zi
            counts[iy, ix] += 1

    print(f"po pętli...")

    # Uśrednianie tam, gdzie było więcej punktów
    mask = counts > 0
    grid_z[mask] = grid_z[mask] / counts[mask]

    # Współrzędne osi
    xi = np.linspace(x_min, x_max, grid_size_x)
    yi = np.linspace(y_min, y_max, grid_size_y)

    return grid_z, xi, yi

def interpolate_missing(grid_z, xi, yi, x_data, y_data, z_data):
    # Siatka jako współrzędne (X, Y)
    grid_x, grid_y = np.meshgrid(xi, yi)

    # Maska brakujących danych
    mask_nan = np.isnan(grid_z)

    # Punkty do interpolacji – tylko tam, gdzie brakuje danych
    interp_points = np.column_stack((grid_x[mask_nan], grid_y[mask_nan]))

    # Interpolacja
    interpolated = griddata(
        points=(x_data, y_data),
        values=z_data,
        xi=interp_points,
        method='linear'
    )

    # Wstawienie wartości tylko w brakujących miejscach
    grid_z[mask_nan] = interpolated

    return grid_z

def interpolate_grid_in_chunks(x, y, z, xi, yi, block_size=512, method='linear'):
    from scipy.interpolate import griddata
    import numpy as np

    grid_z = np.full((len(yi), len(xi)), np.nan, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xi, yi)

    xg = grid_x
    yg = grid_y

    for y0 in range(0, len(yi), block_size):
        for x0 in range(0, len(xi), block_size):
            y1 = min(y0 + block_size, len(yi))
            x1 = min(x0 + block_size, len(xi))

            gx_block = xg[y0:y1, x0:x1]
            gy_block = yg[y0:y1, x0:x1]

            coords = np.column_stack((gx_block.ravel(), gy_block.ravel()))
            interp_block = griddata(
                (x, y), z, coords, method=method
            ).reshape(gx_block.shape)

            grid_z[y0:y1, x0:x1] = interp_block

    return grid_z

def interpolate_missing_pointwise(grid_z, xi, yi, x_data, y_data, z_data, batch_size=1000):
    from scipy.interpolate import LinearNDInterpolator

    # Tworzenie interpolatora
    interp = LinearNDInterpolator(list(zip(x_data, y_data)), z_data)

    # Siatka
    grid_x, grid_y = np.meshgrid(xi, yi)
    mask_nan = np.isnan(grid_z)
    coords = np.column_stack((grid_x[mask_nan], grid_y[mask_nan]))

    print(f"Interpoluję {len(coords)} brakujących punktów w paczkach po {batch_size}...")

    # Paczkowa interpolacja
    for i in range(0, len(coords), batch_size):
        batch = coords[i:i + batch_size]
        result = interp(batch)
        grid_z[np.where(mask_nan)[0][i:i + batch_size]] = result

    return grid_z


def mask_outside_convex_hull(grid_z, xi, yi, x_data, y_data):
    # Tworzenie obwiedni
    points = np.column_stack((x_data, y_data))
    hull = ConvexHull(points)
    polygon = Path(points[hull.vertices])

    # Siatka jako punkty (2D)
    grid_x, grid_y = np.meshgrid(xi, yi)
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    # Test przynależności
    inside = polygon.contains_points(grid_points).reshape(grid_z.shape)

    # Zamaskuj punkty spoza obwiedni
    grid_z[~inside] = np.nan

    return grid_z


src = "source_data/40a.dat.csv"
dst = "source_data/scan1.npz"

# Załaduj dane
df = pd.read_csv(src, sep=';', header=None, names=['x', 'y', 'z'])

df = df.dropna()

# Automatyczne określenie kroku
dx = np.diff(np.sort(np.unique(df['x'])))
dy = np.diff(np.sort(np.unique(df['y'])))
pixel_size_x = np.median(dx[dx > 0])
pixel_size_y = np.median(dy[dy > 0])

print(f"Rozmiar piksela: {pixel_size_x} x {pixel_size_y}")

pixel_size_x = 1.38
pixel_size_y = 1.38

print(f"Rozmiar piksela: {pixel_size_x} x {pixel_size_y}")

print(f"buduje siatke...")
# 1. Budowanie siatki z bezpośrednim przypisaniem
grid_z, xi, yi = build_grid_direct(df, pixel_size_x, pixel_size_y)

print(f"maskuje punkty poza obiektem...")
x_data, y_data = df['x'].values, df['y'].values
grid_z = mask_outside_convex_hull(grid_z, xi, yi, x_data, y_data)

# print(f"interpoluje brakujace wartosci...")
z_data = df['z'].values
# grid_z = interpolate_missing(grid_z, xi, yi, x_data, y_data, z_data)

print("Interpolacja blokowa...")
#grid_z = interpolate_grid_in_chunks(x_data, y_data, z_data, xi, yi, block_size=512, method='linear')
grid_z = interpolate_missing_pointwise(grid_z, xi, yi, x_data, y_data, z_data, batch_size=1000)

# Zapis
np.savez(dst, grid=grid_z)

