
from skimage import data

from imageio import imwrite

import matplotlib.pyplot as pp

import numpy as np


# Save camera image

imwrite('fotografo.png', data.camera())


# Show camera image with colorbar

pp.figure()

camera = data.camera()

pp.imshow(camera, cmap='gray')

pp.colorbar()


# Show camera image with colorbar, without axis and the hist

fig, (ax1, ax2) = pp.subplots(1, 2, figsize=(8, 3))

ax1.imshow(camera, cmap='gray')

ax1.axis('off')

# pp.colorbar(pp.cm.ScalarMappable(cmap='gray').set_array([0,1]))

normalized = lambda ax: lambda image: ax.hist(image.flatten(), bins=256,
         weights=np.ones(image.ravel().shape) / float(image.size))

normalized(ax2)(camera)


# Cut camera image in four parts

def cut(image, rows, cols):
    '''
        Cut image into four quadrants

        Parameters
        ----------

        image:
            Image numpy arrange

        rows:
            Quantity of rows

        rows:
            Quantity of columns

        Usage
        -----

        >>> cut(skimage.data.camera(), 8, 8)

        Return
        ------

        Returns a list containing the four quadrants
    '''

    return [
        image[0:rows // 2, 0:cols // 2],  # t-l
        image[0:rows // 2, cols // 2:],  # t-r
        image[rows // 2:, 0:cols // 2],  # b-l
        image[rows // 2:, cols // 2:]  # b-r
    ]


quadrants = cut(camera, camera.shape[0], camera.shape[1])

for count, item in enumerate(quadrants, 1):
    imwrite(f'fotografo_{count}.png', item)

    # pp.figure()

    # pp.imshow(item, cmap='gray')


# Show camera, hist and the quadrants in one figure

fig, ax = pp.subplots(3, 2)

ax[0, 0].imshow(camera, cmap='gray')

normalized(ax[0, 1])(camera)

count = 0

for r in range(1, 3):
    for c in range(0, 2):
        ax[r, c].imshow(quadrants[count], cmap='gray')

        count += 1


# Plot everything

pp.show()
