import os

from imageio import imread, imsave

import random

import numpy as np

# Open all images generated on previous scripts

png = lambda item: item if item.endswith('.png') else None

names = list(filter(png, os.listdir()))

imgs = [imread(os.path.join(os.getcwd(), image)) for image in names]

# Get meta data from images

meta = []

for name, image in zip(names, imgs):
    cache = [
        image.shape[0],  # Lines
        image.shape[1],  # Cols
        image.max(),  # Max intensity
        image.min(),  # Min intensity
        image.mean(),  # Mean intensity
        # np.median(image),  # Median intensity
        image.std(),  # Standard deviation
        np.sum(image > image.mean()),  # Pixels large than std
        np.sum(image <= image.mean()),  # Pixels smaller than std
    ]
    
    meta.append(cache)

    with open(f'{name[:-4]}.txt', 'w') as f:
        f.write(str(cache))

    if cache[-2] > cache[-1]:
        imsave(f'./light/{name}', image)
    else:
        imsave(f'./dark/{name}', image)
