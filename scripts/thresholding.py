import numpy as np
import sys
import matplotlib.pyplot as plt

arguments = sys.argv[1:]
print("Creating threshold for image ", arguments[0])

IMAGE = int(arguments[0])
PATH_FIG=f'data/example_tiles_complete/12.5cm-Aerial-Photo/change-tiles/image_{IMAGE}'
PATH_FIG_THRESHOLD=f'data/example_tiles_complete/12.5cm-Aerial-Photo/change-tiles/image_{IMAGE}_threshold'


big_array = np.load(f'{PATH_FIG_THRESHOLD}/array.npy')
print(big_array.max())
print(big_array.min())

plt.imshow(big_array, interpolation='none')
plt.savefig(f'{PATH_FIG_THRESHOLD}/all.png', cmap='binary', dpi=300)


for id, i in enumerate([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]):
    # threshold = np.where(big_array > i, 1, 0)  # get the index of the max log-probability
    threshold = 1.0 * (big_array > i)
    # print(thres)
    np.save(f'{PATH_FIG_THRESHOLD}/threshold{id}', threshold)
    plt.imshow(threshold, interpolation='none')
    plt.savefig(f'{PATH_FIG_THRESHOLD}/threshold{id}.png', cmap='binary', dpi=300)
    # exit()