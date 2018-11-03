# Run : python l04_e01.py 1, python l04_e01.py 2
 
from skimage import img_as_float

from os import listdir, path

from imageio import imread

import numpy as np

from sklearn import preprocessing, model_selection

import matplotlib.pyplot as pp

import seaborn as sn

from utils import plot, morphologic, models

from sys import argv

# Settings

folderName = 'mpeg7'  # flavia, mpeg7

if len(argv) > 1:
    if argv[1] in ['1', '2']:
        if argv[1] == '1':
            folderName = 'mpeg7'
        elif argv[1] == '2':
            folderName = 'flavia'

rgb = False

if [folder for folder in ['flavia'] if folderName == folder]:
    rgb = True

# Load images

folderPath = './data/' + folderName

folder = {}

for className in listdir(folderPath):
    folder.update(
        {
            className: {
                image[:-4]:
                        img_as_float(imread(path.join(folderPath, className, image)))[:, :, 0]
                    if rgb else
                        img_as_float(imread(path.join(folderPath, className, image)))
                    for image in sorted(listdir(path.join(folderPath, className)))
            }
        }
    )

# Get morphologic properties

properties, propertiesCache, labelsCache = morphologic(folder)

folderData = np.vstack(propertiesCache)

folderLabels = preprocessing.LabelEncoder().fit_transform(labelsCache)

# Save data

# np.savetxt(
#     folderName + '.txt',
#     np.c_[folderData, folderLabels],
#     fmt='%10.5f',
# )

# Standard scaler (transformada normal)

folderData = preprocessing.StandardScaler().fit_transform(folderData)

# Split data

x_train, x_test, y_train, y_test = model_selection.train_test_split(
    folderData,
    folderLabels,
    test_size=0.25
)

# Plot normal distribution

np.random.seed(0)

folderDataNormalized = preprocessing.minmax_scale(folderData)

mu, sigma = folderDataNormalized.mean(), folderDataNormalized.std()

sn.set()

normalDistribution = sn.distplot(np.random.normal(mu, sigma, 1000))

normalDistribution.set_title('Normal Distribution')

# Classify data

predicted = models(x_train, y_train, x_test)

# Plot confusion matrix

fig, ax = pp.subplots(2, 3, figsize=(9, 6))

predictedMetrics = {}

predictedMetrics.update(
    plot(ax, 0, list(predicted.items())[:3], y_test)
)

predictedMetrics.update(
    plot(ax, 1, list(predicted.items())[3:], y_test)
)

pp.show()
