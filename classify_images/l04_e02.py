from os import path, listdir

from sklearn import preprocessing, model_selection

from skimage import img_as_float

from imageio import imread

from utils import plot, morphologic, models

import numpy as np

from sys import argv

import matplotlib.pyplot as pp

# Settings

folderName = 'mpeg7_2'  # flavia_2, mpeg7_2

if len(argv) > 1:
    if argv[1] in ['1', '2']:
        if argv[1] == '1':
            folderName = 'mpeg7_2'
        elif argv[1] == '2':
            folderName = 'flavia_2'

rgb = False

if [folder for folder in ['flavia_2'] if folderName == folder]:
    rgb = True

# Load images

folderPath = './data/' + folderName

folder = {}

for imageName in sorted(listdir(folderPath)):
    folder.update(
        {
            imageName[:-4]:
                img_as_float(imread(path.join(folderPath, imageName)))[:, :, 0] if rgb else img_as_float(imread(path.join(folderPath, imageName)))
        }
    )

# Get morphologic properties

properties, propertiesCache, labelsCache = morphologic(folder, simple=True)

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