from scipy.interpolate import griddata
import numpy as np

def convert_to_full(src,src_grid,dst):
    # Załaduj dane
    data = np.load(src)
    x=data['x_data']
    y=data['y_data']
    z=data['z_data']

    data = np.load(src_grid)
    grid = data['grid']
    pixel_size_x=data['px_x']
    pixel_size_y=data['px_y']
    xi=data['xi']
    yi=data['yi']

    print(f"Rozmiar piksela: {pixel_size_x} x {pixel_size_y}")

    grid_size_y, grid_size_x = grid.shape

    print(f"Rozmiar siatki: {grid_size_x} x {grid_size_y}")

    # Wymiary siatki
    x_min = xi[0]
    x_max = xi[-1]
    y_min = yi[0]
    y_max = yi[-1]

    print(f"x_min: {x_min}, y_min: {y_min}")
    print(f"x_max: {x_max}, y_min: {y_max}")

    grid_x, grid_y = np.meshgrid(xi, yi)

    # Szybka interpolacja
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='nearest')

    np.savez(dst, grid=grid_z, xi=xi, yi=yi, px_x=pixel_size_x, px_y=pixel_size_y)

src = "source_data/40a.dat.npz"
src_grid = "source_data/scan1_org.npz"
dst = "source_data/scan1_full_interp.npz"

convert_to_full(src,src_grid,dst)

src = "source_data/40b.dat.npz"
src_grid = "source_data/scan2_org.npz"
dst = "source_data/scan2_full_interp.npz"

convert_to_full(src,src_grid,dst)
