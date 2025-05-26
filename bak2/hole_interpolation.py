from scipy.interpolate import griddata
import numpy as np
from skimage.segmentation import flood

def test_org(src,src_grid,dst):
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

    tst = np.isnan( grid )

    print("Liczba nan: ", np.count_nonzero(tst))

    # tst – to jest Twoja maska bool (np. tst = np.isnan(grid))
    # Wypełnij obszar zaczynając od (0, 0)
    filled = flood(tst, seed_point=(50, 50))

    # Usuń ten obszar z maski (czyli ustaw na False)
    tst[filled] = False

    print("Liczba nan: ", np.count_nonzero(tst))

    a,b = grid.shape

    filled = flood(tst, seed_point=(50, b-50))

    # Usuń ten obszar z maski (czyli ustaw na False)
    tst[filled] = False

    print("Liczba nan: ", np.count_nonzero(tst))

    filled = flood(tst, seed_point=(a-50, 50))

    # Usuń ten obszar z maski (czyli ustaw na False)
    tst[filled] = False

    print("Liczba nan: ", np.count_nonzero(tst))

    filled = flood(tst, seed_point=(a-50, b-50))

    # Usuń ten obszar z maski (czyli ustaw na False)
    tst[filled] = False

    print("Liczba nan: ", np.count_nonzero(tst))

    grid_x, grid_y = np.meshgrid(xi, yi)
    # 2. Wyciągamy tylko współrzędne punktów, gdzie tst == True
    interp_points = np.column_stack((grid_x[tst], grid_y[tst]))

    # 3. Interpolujemy TYLKO te punkty
    interp_values = griddata((x, y), z, interp_points, method='nearest')

    # 4. Wstawiamy je z powrotem do grid
    grid[tst] = interp_values

    np.savez(dst, grid=grid, xi=xi, yi=yi, px_x=pixel_size_x, px_y=pixel_size_y)

    # import matplotlib.pyplot as plt

    # plt.imshow(grid, cmap='gray', origin='lower')
    # plt.title("Maska NaN (True = NaN)")
    # plt.show()


src = "source_data/40a.dat.npz"
src_grid = "source_data/scan1_org.npz"
dst = "source_data/scan1_interp.npz"

test_org(src,src_grid,dst)

src = "source_data/40b.dat.npz"
src_grid = "source_data/scan2_org.npz"
dst = "source_data/scan2_interp.npz"

test_org(src,src_grid,dst)
