
import os

# import re

from imageio import imread, imsave

import random

import numpy as np

# Open all images and get their names

# imgs = list(filter(re.compile('([-\w]+\.(?:jpg|gif|png))').match, listdir()))


def png(item):
    '''
        Checks if string ends with png extension

        Parameters
        ----------

        item:
            File name

        Usage
        -----

        >>> png('item.png')

        Return
        ------

        File's name if the same endswith png extension
    '''

    if item.endswith('.png'):
        return item


names = list(filter(png, os.listdir()))

imgs = [imread(os.path.join(os.getcwd(), image)) for image in names]

# Get meta data from images

meta = []

for name, image in zip(names, imgs):
    cache = [
        image.shape[0],  # lines
        image.shape[1],  # cols
        image.max(),  # max intensity
        image.min(),  # min intensity
        image.mean(),  # mean intensity
        # np.median(image),  # median intensity
        image.std(),  # standard deviation
        np.sum(image > image.mean()),  # pixels large than std
        np.sum(image <= image.mean()),  # pixels smaller than std
    ]

    print(cache)

    meta.append(cache)

    with open(f'{name[:-4]}.txt', 'w') as f:
        f.write(str(cache))

    if cache[-2] > cache[-1]:
        imsave(f'./light/{name}', image)
    else:
        imsave(f'./dark/{name}', image)
