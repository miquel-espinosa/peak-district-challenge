import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import sys

arguments = sys.argv[1:]
print("Merging patches for image", arguments[0])

IMAGE = int(arguments[0])
PATH_FIG=f'data/example_tiles_complete/12.5cm-Aerial-Photo/change-tiles/image_{IMAGE}'
PATH_FIG_THRESHOLD=f'data/example_tiles_complete/12.5cm-Aerial-Photo/change-tiles/image_{IMAGE}_threshold'

if not os.path.exists(f'{PATH_FIG_THRESHOLD}'):
    os.makedirs(f'{PATH_FIG_THRESHOLD}')

NUM_PATCHES = 62
RESOLUTION = 128
DIMENSION = NUM_PATCHES*RESOLUTION

big_array = np.zeros((DIMENSION, DIMENSION))

for idx, i in enumerate(range(0, DIMENSION, RESOLUTION)):
    for idy, j in enumerate(range(0, DIMENSION, RESOLUTION)):
        # print(i,i+RESOLUTION, j, j+RESOLUTION)
        big_array[i:i+RESOLUTION,j:j+RESOLUTION] = np.squeeze(joblib.load(f'{PATH_FIG}/{idx*NUM_PATCHES+idy}.joblib'))

np.save(f'{PATH_FIG_THRESHOLD}/array', big_array)

