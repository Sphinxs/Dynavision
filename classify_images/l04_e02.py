from os import path, listdir

from sklearn import preprocessing, model_selection

from imageio import imread

from utils import plot, morphologic, models, formatFigure

import numpy as np

from sys import argv

import matplotlib.pyplot as pp

# Folder settings

folderName = ''

if len(argv) >= 2:
    if argv[1].lower() in ['flavia_2', 'mpeg7_2']:
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

for imageName in sorted(listdir(folderPath)):
    folder.update({
        imageName: formatFigure(
            imread(path.join(folderPath, imageName))
        )
    })

# Morphologic properties

properties, labels = morphologic(folder, simple=True)

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

# Classify data

predicted = models(x_train, y_train, x_test)

# Confusion matrix

pp.rcParams['toolbar'] = 'None'

fig, ax = pp.subplots(2, 4, figsize=(10, 6))

fig.canvas.set_window_title(folderName.capitalize().split('_'))

plot(ax, 0, list(predicted.items())[:4], y_test)

plot(ax, 1, list(predicted.items())[4:], y_test)

pp.show()
