
from skimage import data, img_as_float, exposure

from imageio import imsave

import matplotlib.pyplot as pp

import numpy as np


# Load coin image and save

coins = data.coins()

imsave('moedas.png', coins)

coins = img_as_float(coins)

processed = [coins]


# Image contrast (intensity) widening

processed.append(exposure.rescale_intensity(coins))

processed.append(exposure.rescale_intensity(coins, (.2, .7)))

imsave('l02_e04_ac.png', processed[1])

imsave('l02_e04_ac27.png', processed[2])


# Image equalization

processed.append(exposure.equalize_hist(coins))

imsave('l02_e04_eq.png', processed[3])


# Plots all exposure

fix, ax = pp.subplots(4, 2)

for count, image in enumerate(processed):
    ax[count, 0].imshow(image, cmap='gray')

    ax[count, 0].axis('off')

    ax[count, 1].hist(image.flatten(), bins=256, weights=np.ones(image.ravel().shape) / float(image.size))

pp.show()