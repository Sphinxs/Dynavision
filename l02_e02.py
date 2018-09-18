
from imageio import imread, imsave

from skimage import img_as_float

import matplotlib.pyplot as pp

import numpy as np

import random

from scipy import ndimage


# Load image from disk

camera = img_as_float(imread('fotografo.png'))


# Split images in 4 lines and 4 cols (16 squads)

squads = []


def split(img, rows, cols):
    '''
        Split array in n rows and columns

        Parameters
        ----------

        img:
            Arbitrary Numpy array

        rows:
            Number of rows to split the array

        cols:
            Number of cols to split the array

        Usage
        -----

        >>> split(skimage.data.camera(), 4, 4)

        Return
        ------

        Python list containing the subsets
    '''

    cache = []

    try:
        img_r = img.shape[0]

        img_c = img.shape[1]
    except Exception as e:
        raise Exception(
            f'\nInform a \033[31mNumpy\033[37m array\n\n{str(e)}\n')

    # Count | Next

    for c, n in zip(range(0, img_r + 1, img_r // rows), range(img_r // rows, img_r + 1, img_r // rows)):

        # Interator | Forward

        for i, f in zip(range(0, img_c + 1, img_c // cols), range(img_c // cols, img_c + 1, img_r // cols)):
            cache.append(img[c:n, i:f])

    return cache


# Generate 20 values to rotate the squads

random.seed(42)  # 42 Is the answer for everything

r_values = random.sample(range(20, 341), 16)


# Save images with G º

squads = split(camera, 4, 4)

count = 0

for img, degrees in zip(squads, r_values):
    imsave(f'fotografo_{degrees:.2f}.png', ndimage.rotate(squads[count], r_values[count], reshape=False))

    count += 1

    # imsave(f'fotografo_{degrees:.2f}.png', img)


# Plot all squads

fig1, ax1 = pp.subplots(4, 4)

fix2, ax2 = pp.subplots(4, 4)

fix3, ax3 = pp.subplots(4, 4)

count = 0

for line in range(0, 4):
    for col in range(0, 4):
        # Normal image

        ax1[line, col].imshow(squads[count], cmap='gray')

        ax1[line, col].axis('off')

        # Image rotated

        ax2[line, col].imshow(ndimage.rotate(
            squads[count], r_values[count], reshape=False), cmap='gray')

        ax2[line, col].axis('off')

        # Normal image hist

        ax3[line, col].hist(squads[count].flatten(), bins=256, weights=np.ones(
            squads[count].ravel().shape) / float(squads[count].size))

        count += 1

pp.show()
