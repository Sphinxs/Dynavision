from imageio import imread, imsave

from skimage import img_as_float

import matplotlib.pyplot as pp

import numpy as np

import random

from scipy import ndimage

# Load camera image from disk and convert to float

camera = img_as_float(imread('photographer.png'))

# Split image in 4 rows and 4 cols (16 quadrants)

quadrants = []

def split(img, rows, cols):
    '''
        Split image in n rows and n columns

        Parameters
        ----------

        img: numpy.array
            Image matrix to split

        rows: int
            Number of rows to split

        cols: int
            Number of cols to split

        Usage
        -----

        >>> split(skimage.data.camera(), 4, 4)

        Return
        ------

        List containing n quadrants from original image
    '''

    cache = []

    try:
        img_r = img.shape[0]

        img_c = img.shape[1]
    except Exception as e:
        raise Exception(f'\nInform a \033[31mNumpy\033[37m array\n\n{str(e)}\n')

    for c, n in zip(range(0, img_r + 1, img_r // rows), range(img_r // rows, img_r + 1, img_r // rows)):
        for i, f in zip(range(0, img_c + 1, img_c // cols), range(img_c // cols, img_c + 1, img_r // cols)):
            cache.append(img[c:n, i:f])

    return cache

# Generate twenty values to rotate quadrants

random.seed(42)

r_values = random.sample(range(20, 341), 16)

# Slipt image and save it's quadrants

quadrants = split(camera, 4, 4)

count = 0

for img, degrees in zip(quadrants, r_values):
    imsave(f'photographer_{degrees:.2f}.png', ndimage.rotate(quadrants[count], r_values[count], reshape=False))

    count += 1

    imsave(f'photographer_{degrees:.2f}.png', img)

# Show camera image and it's quadrants and histograms

fig1, ax1 = pp.subplots(4, 4)

fix2, ax2 = pp.subplots(4, 4)

fix3, ax3 = pp.subplots(4, 4)

count = 0

for line in range(0, 4):
    for col in range(0, 4):
        # Original image

        ax1[line, col].imshow(quadrants[count], cmap='gray')

        ax1[line, col].axis('off')

        # Image rotated

        ax2[line, col].imshow(ndimage.rotate(quadrants[count], r_values[count], reshape=False), cmap='gray')

        ax2[line, col].axis('off')

        # Original image hist

        ax3[line, col].hist(quadrants[count].flatten(), bins=256, weights=np.ones(quadrants[count].ravel().shape) / float(quadrants[count].size))

        count += 1

pp.show()