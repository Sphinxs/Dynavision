from skimage import data

from imageio import imwrite

import matplotlib.pyplot as pp

import numpy as np

# Load camera image and save it

camera = data.camera()

imwrite('photographer.png', camera)

# Show camera image with color bar

pp.figure()

pp.imshow(camera, cmap='gray')

pp.colorbar()

# Show original camera and it's histogram

fig, (ax1, ax2) = pp.subplots(1, 2, figsize=(8, 3))

ax1.imshow(camera, cmap='gray')

ax1.axis('off')

normalized = lambda ax: lambda image: ax.hist(
    image.flatten(),
    bins=256,
    weights=np.ones(image.ravel().shape) / float(image.size)
)

normalized(ax2)(camera)

# Split image in four quadrants and save it

def split(image, rows, cols):
    '''
        Split image in four quadrants

        Parameters
        ----------

        image: numpy.array
            Image matrix to split

        rows: int
            Amount of rows from image

        cols: int
            Amount of columns from image

        Usage
        -----

        >>> split(skimage.data.camera(), 4, 4)

        Return
        ------

        List containing four quadrants from original image
    '''

    return [
        image[0:rows // 2, 0:cols // 2],  # t-l
        image[0:rows // 2, cols // 2:],  # t-r
        image[rows // 2:, 0:cols // 2],  # b-l
        image[rows // 2:, cols // 2:]  # b-r
    ]

quadrants = split(camera, camera.shape[0], camera.shape[1])

for count, item in enumerate(quadrants, 1):
    imwrite(f'photographer_{count}.png', item)

# Show camera image, it's hist and image splited in four quadrants

fig, ax = pp.subplots(3, 2)

ax[0, 0].imshow(camera, cmap='gray')

normalized(ax[0, 1])(camera)

count = 0

for r in range(1, 3):
    for c in range(0, 2):
        ax[r, c].imshow(quadrants[count], cmap='gray')

        count += 1

pp.show()
