from skimage import measure, img_as_float

from skimage.color import rgb2gray

from os import listdir, path

from imageio import imread

from math import pi

import numpy as np

from sklearn import preprocessing, model_selection, metrics

import matplotlib.pyplot as pp

import seaborn as sn

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn import svm

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from scipy.fftpack import fft, ifft

from json import dumps

# Settings

# It's necessary to have more than one class because the algorithms require it

folderName = 'flavia'  # mpeg7, flavia

# If the image is RGB it's necessary to convert the 3D array to 2D array

rgb = True

# Load Dataset

folderPath = './data/' + folderName

folder = {}

for className in listdir(folderPath):
    folder.update(
        {
            className: {
                image[:-4]:
                    img_as_float(imread(path.join(folderPath, className, image)))[:,:,0] if rgb else img_as_float(imread(path.join(folderPath, className, image)))
                    for image in sorted(listdir(path.join(folderPath, className)))
            }
        }
    )

# Get morphological properties (data and labels) and save in a dataset

folderProperties = {}

folderPropertiesCache = []

folderLabelsCache = []

for className, imagesDict in folder.items():
    folderProperties.update({className: {}})

    for imageName, imageData in imagesDict.items():
        # Get labels

        imageLabeled = measure.label(imageData, background=0)

        # Get properties

        for prop in measure.regionprops(imageLabeled, imageData, coordinates='rc'):
            folderProperties[className].update({
                imageName: [
                    prop.area,
                    prop.eccentricity,
                    prop.mean_intensity,
                    prop.solidity,
                    # (4 * pi * prop.area) / prop.perimeter ** float(2)  # Circularity
                ]
            })

            folderPropertiesCache.append(
                folderProperties[className][imageName])

            folderLabelsCache.append(className)
            
            print('Loop', '\n')

folderData = np.vstack(folderPropertiesCache)

# Generate labels

lp = preprocessing.LabelEncoder()

lp.fit(folderLabelsCache)

folderLabels = lp.transform(folderLabelsCache)

folderLabelsNames = lp.classes_

# Save data and labels

np.savetxt(
    folderName + '.txt',
    np.c_[folderData, folderLabels],
    delimiter=',',
    fmt='%10.3f',
    header='Area, eccentricity, mean_intensity, solidity, circularity, label'
)

# Fourier transformation

fftY = fft(folderData)

invY = ifft(fftY)

# folderData = invY

# Split dataset

x_train, x_test, y_train, y_test = model_selection.train_test_split(
    folderData,
    folderLabels,
    test_size=0.25
)

# Normal distribution

np.random.seed(0)

folderDataNormalized = preprocessing.minmax_scale(folderData)

mu, sigma = folderDataNormalized.mean(), folderDataNormalized.std()

sn.set()

ax = sn.distplot(np.random.normal(mu, sigma, 1000))

ax.set_title('Normal Distribution')

pp.show()

# Classify

predicted = {}

# Gaussian Process Classifier

gaussian = GaussianProcessClassifier(1.0 * RBF(1.0))

gaussian.fit(x_train, y_train)

predicted.update({
    'Gaussian Process': gaussian.predict(x_test)
})

# Knn

knn = KNeighborsClassifier(n_neighbors=4)

knn.fit(x_train, y_train)

predicted.update({
    'K-Nearest Neighbors': knn.predict(x_test)
})

# Gaussian Naive Bayes

bayes = GaussianNB()

bayes.fit(x_train, y_train)

predicted.update({
    'Naive Bayes': bayes.predict(x_test)
})

# Decision Tree

tree = DecisionTreeClassifier()  # max_depth=5

tree.fit(x_train, y_train)

predicted.update({
    'Decision Tree': tree.predict(x_test)
})

# Random Forest

random = RandomForestClassifier(n_estimators=20)

random.fit(x_train, y_train)

predicted.update({
    'Random Forest': random.predict(x_test)
})

# Svm

svm = svm.SVC(gamma='scale')  # 1 / (n_features * X.std())

svm.fit(x_train, y_train)

predicted.update({
    'Suport Vector Machine': svm.predict(x_test)
})

# Get metrics and plot confusion matrix


def plotLine(ax, row, figs):
    '''
        Plot n figures in a row

        Parameters
        ----------

        ax:
            Subplot to plot the figures

        row:
            Row on subplot to plot the figures

        figs:
            List of figures (tuples -> key, value)

        Returns
        -------

        Metrics from figures received by the function

        Usage
        -----

        >>> None
    '''

    counter = 0

    cache = {}

    for k, v in figs:
        confusion = metrics.confusion_matrix(y_test, v)

        sn.heatmap(confusion, annot=True, ax=ax[row, counter], cbar=False)

        ax[row, counter].set_title(k)

        ax[row, counter].axis('off')

        counter += 1

        cache.update({
            k: {
                'accuracy': metrics.accuracy_score(y_test, v)
            }
        })

        print(
            f'\n{k} - \033[34mReport\033[m:\n\n{metrics.classification_report(y_test, v)}', end='\n'
        )

    return cache


fig, ax = pp.subplots(2, 3, figsize=(9, 6))

predictedMetrics = {}

predictedMetrics.update(
    plotLine(ax, 0, list(predicted.items())[:3])
)

predictedMetrics.update(
    plotLine(ax, 1, list(predicted.items())[3:7])
)

pp.show()

# List of models

print('\n', dumps(
    dict(
        reversed(
            sorted(
                predictedMetrics.items(),
                key=lambda item: item[1]['accuracy']
            )
        )
    ),
    sort_keys=False,
    indent=3,
    separators=('\n', ' : ')
), '\n')
