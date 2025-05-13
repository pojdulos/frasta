from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
from matplotlib.path import Path
import numpy as np

def mask_outside_convex_hull(grid_full, xi, yi, x_data, y_data):
    # Tworzenie obwiedni
    points = np.column_stack((x_data, y_data))
    hull = ConvexHull(points)
    polygon = Path(points[hull.vertices])

    # Siatka jako punkty (2D)
    grid_x, grid_y = np.meshgrid(xi, yi)
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    # Test przynależności
    inside = polygon.contains_points(grid_points).reshape(grid_full.shape)

    # Zamaskuj punkty spoza obwiedni
    grid_full[~inside] = np.nan

    return grid_full

def convert_to_scan(src_org,src_full,dst):
    data = np.load(src)
    x_data=data['x_data']
    y_data=data['y_data']
    #z_data=data['z_data']
    
    data = np.load(src_full)
    grid_full = data['grid']
    pixel_size_x=data['px_x']
    pixel_size_y=data['px_y']
    xi=data['xi']
    yi=data['yi']

    print(f"Rozmiar piksela: {pixel_size_x} x {pixel_size_y}")

    grid_size_y, grid_size_x = grid_full.shape

    print(f"Rozmiar siatki: {grid_size_x} x {grid_size_y}")

    # Wymiary siatki
    x_min = xi[0]
    x_max = xi[-1]
    y_min = yi[0]
    y_max = yi[-1]

    print(f"x_min: {x_min}, y_min: {y_min}")
    print(f"x_max: {x_max}, y_min: {y_max}")

    grid_z = mask_outside_convex_hull(grid_full, xi, yi, x_data, y_data)

    np.savez(dst, grid=grid_z, xi=xi, yi=yi, px_x=pixel_size_x, px_y=pixel_size_y)

src = "source_data/40a.dat.npz"
src_full = "source_data/scan1_full_interp.npz"
dst = "source_data/scan1.npz"

convert_to_scan(src,src_full,dst)

src = "source_data/40b.dat.npz"
src_full = "source_data/scan2_full_interp.npz"
dst = "source_data/scan2.npz"

convert_to_scan(src,src_full,dst)
