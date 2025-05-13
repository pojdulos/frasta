import pandas as pd
import numpy as np

def convert(src,dst_src,dst_grid):
    # Załaduj dane
    df = pd.read_csv(src, sep=';', header=None, names=['x', 'y', 'z'])

    df = df.dropna()

    # Automatyczne określenie kroku
    dx = np.diff(np.sort(np.unique(df['x'])))
    dy = np.diff(np.sort(np.unique(df['y'])))
    
    pixel_size_x = np.median(dx[dx > 0]).round(2)
    pixel_size_y = np.median(dy[dy > 0]).round(2)

    print(f"Rozmiar piksela: {pixel_size_x} x {pixel_size_y}")

    pixel_size_x = 1.38
    pixel_size_y = 1.38

    print(f"Rozmiar piksela: {pixel_size_x} x {pixel_size_y}")

    print(f"buduje siatke...")
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

    print("counts max przed pętlą:", counts.max())

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

    print("counts max po pętli:", counts.max())
    
    print("liczba counts>1:", np.count_nonzero(counts > 1))

    mask_dup = (counts > 1)

    print("mask shape:", mask_dup.shape)
    print("mask nonzero count:", np.count_nonzero(mask_dup))

    grid_z[mask_dup] = grid_z[mask_dup] / counts[mask_dup]

    # Współrzędne osi
    _xi = np.linspace(x_min, x_max, grid_size_x)
    _yi = np.linspace(y_min, y_max, grid_size_y)


    np.savez(dst_src,
        x_data=x,    # 1D array
        y_data=y,
        z_data=z )

    np.savez(dst_grid,
        grid=grid_z,
        x_min=x_min,
        y_min=y_min,
        px_x=pixel_size_x,
        px_y=pixel_size_y,
        xi=_xi,
        yi=_yi )




src = "source_data/40a.dat.csv"
dst_src = "source_data/40a.dat.npz"
dst_grid = "source_data/scan1_org.npz"

convert(src,dst_src,dst_grid)

src = "source_data/40b.dat.csv"
dst_src = "source_data/40b.dat.npz"
dst_grid = "source_data/scan2_org.npz"

convert(src,dst_src,dst_grid)

