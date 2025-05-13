import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, shift
import matplotlib.patches as patches
import os
import pandas as pd


df_a = pd.read_csv('source_data/40a.dat.csv', sep=';', header=None, quotechar='"')
df_b = pd.read_csv('source_data/40b.dat.csv', sep=';', header=None, quotechar='"')

scan_a = np.array(df_a.values.tolist(), dtype=np.float32)
scan_b = np.array(df_b.values.tolist(), dtype=np.float32)

np.save("source_data/scan_a.npy", scan_a)
np.save("source_data/scan_b.npy", scan_b)


plt.imshow(scan_a, cmap='gray')
plt.title('Scan 1')
plt.colorbar()
plt.show()

plt.imshow(scan_a, cmap='gray')
plt.title('Scan 2')
plt.colorbar()
plt.show()
