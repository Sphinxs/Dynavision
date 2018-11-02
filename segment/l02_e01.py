from skimage import data

from imageio import imwrite

import matplotlib.pyplot as pp

import numpy as np

# Save camera image

camera = data.camera()

imwrite('fotografo.png', camera)

# Show camera image with color bar

pp.figure()

pp.imshow(camera, cmap='gray')

pp.colorbar()

# Show hist and camera image with colorbar and without axis

fig, (ax1, ax2) = pp.subplots(1, 2, figsize=(8, 3))

ax1.imshow(camera, cmap='gray')

ax1.axis('off')

# pp.colorbar(pp.cm.ScalarMappable(cmap='gray').set_array([0,1]))

normalized = lambda ax: lambda image: ax.hist(
    image.flatten(),
    bins=256,
    weights=np.ones(image.ravel().shape) / float(image.size)
)

normalized(ax2)(camera)

# Cut camera image in four parts

def cut(image, rows, cols):
    '''
        Cut image into four quadrants

        Parameters
        ----------

        image:
            Image to be splited

        rows:
            Amount of rows from image

        rows:
            Amount of columns from image

        Usage
        -----

        >>> cut(skimage.data.camera(), 4, 4)

        Return
        ------

        List containing the four quadrants
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

# Show camera, hist and quadrants

fig, ax = pp.subplots(3, 2)

ax[0, 0].imshow(camera, cmap='gray')

normalized(ax[0, 1])(camera)

count = 0

for r in range(1, 3):
    for c in range(0, 2):
        ax[r, c].imshow(quadrants[count], cmap='gray')

        count += 1

# Plot

pp.show()
