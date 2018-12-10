from os import path, getcwd, listdir

from imageio import imread, imsave

from scipy.ndimage.filters import convolve

import numpy as np

from skimage.filters import threshold_otsu

# Read all light images

light = []

for image in listdir(path.join(getcwd(), 'light')):
    if '.png' in image:
        light.append(imread(path.join(getcwd(), 'light', image)))

# Apply median filter

light_median = []

for image in light:
    light_median.append(convolve(image, np.ones((3, 3)) / pow(3, 2)))

# Apply otsu filter

light_otsu = []

light_segmented = []

for indice, image in enumerate(light_median):
    light_otsu.append(threshold_otsu(image))

    light_segmented.append(image > light_otsu[indice])

    imsave(f'./light_seg/l02_e06_{indice}.png', np.invert(image))