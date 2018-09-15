
from imageio import imread

from skimage import img_as_float

import matplotlib.pyplot as pp

import numpy as np


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

    for c, n in zip(range(0, img_r + 1, img_r // rows), range(img_r // rows, img_r + 1, img_r // rows)):
        for i, f in zip(range(0, img_c + 1, img_c // cols), range(img_c // cols, img_c + 1, img_r // cols)):
            cache.append(img[c:n, i:f])

    return cache


for squad in split(camera, 4, 4):
    pp.figure()

    pp.imshow(squad, cmap='gray')

pp.show()
