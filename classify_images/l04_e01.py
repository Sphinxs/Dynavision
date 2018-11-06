from os import path, listdir

from sklearn import preprocessing, model_selection

from imageio import imread

from utils import plot, morphologic, models, formatFigure

import numpy as np

from sys import argv

import seaborn

import matplotlib.pyplot as pp

# Folder settings

folderName = ''

if len(argv) >= 2:
    if argv[1].lower() in ['flavia', 'mpeg7']:
        folderName = argv[1]
    else:
        print('\n\033[31mError\033[m, invalid folder\n')

        raise SystemExit
else:
    print('\n\033[31mError\033[m, inform a folder\n')

    raise SystemExit

folderPath = path.join('./', 'data', folderName)

# Load images

folder = {}

for className in listdir(folderPath):
    folder.update({
        className: {
                imageName:
                    formatFigure(imread(path.join(folderPath, className, imageName)))
                for imageName in sorted(listdir(path.join(folderPath, className)))
            }
    })

## Morphologic properties

properties, labels = morphologic(folder)

folderData = np.vstack(properties)

folderLabels = preprocessing.LabelEncoder().fit_transform(labels)

# Standard scaler

folderData = preprocessing.StandardScaler().fit_transform(folderData)

# Save data

# np.savetxt(
#     ''.join([folderName, '.txt']),
#     np.c_[folderData, folderLabels],
# )

# Split data

x_train, x_test, y_train, y_test = model_selection.train_test_split(
    folderData,
    folderLabels,
    test_size=0.25
)

# Normal distribution

pp.rcParams['toolbar'] = 'None'

np.random.seed(0)

folderDataNormalized = preprocessing.minmax_scale(folderData)

mu, sigma = folderDataNormalized.mean(), folderDataNormalized.std()

seaborn.set()

normalDistribution = seaborn.distplot(np.random.normal(mu, sigma, 1000))

normalDistribution.set_title('Normal Distribution')

# Classify data

predicted = models(x_train, y_train, x_test)  # 'Memory error', try to remove images, algorithms or programs from the memory RAM

# Confusion matrix

fig, ax = pp.subplots(2, 4, figsize=(10, 6))

fig.canvas.set_window_title(folderName.capitalize().split('_'))

plot(ax, 0, list(predicted.items())[:4], y_test)

plot(ax, 1, list(predicted.items())[4:], y_test)

pp.show()