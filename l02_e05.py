
from imageio import imread

from scipy.ndimage.filters import convolve

from skimage import filters, measure, img_as_float

import matplotlib.pyplot as pp

import numpy as np


# Load coins image

coins = img_as_float(imread('moedas.png'))


masks = [(3, 3), (5, 5), (7, 7), (9, 9)]

images = []

therehold = []

segmented = []

label = []

for mask in masks:
    # Mean filter

    cache_m = convolve(coins, np.ones(mask) / pow(mask[0], 2))

    images.append(cache_m)

    # Otsu therehold

    cache_t = filters.threshold_otsu(cache_m)

    therehold.append(cache_t)

    cache_s = cache_m > cache_t

    segmented.append(cache_s)

    # Label images

    cache_l = measure.label(cache_s)

    label.append(cache_l)

    print(
        f'\n\033[31m{cache_l.max()}\033[37m - Elements detected\n')


# Plot the images, hist and the segmented image

fix, ax = pp.subplots(4, 3, figsize=(13, 6))

for count in range(0, 4):
    ax[count, 0].imshow(images[count], cmap='gray')

    ax[count, 0].axis('off')

    ax[count, 0].set_title(''.join(['Mean', str(masks[count])]))

    ax[count, 1].hist(images[count].ravel(), bins=256, weights=np.ones(
        images[count].ravel().shape) / float(images[count].size))

    ax[count, 1].axvline(therehold[count], color='b', linewidth=2)

    ax[count, 2].imshow(segmented[count], cmap='gray')

    ax[count, 2].axis('off')

    ax[count, 2].set_title('Otsu')

pp.show()
