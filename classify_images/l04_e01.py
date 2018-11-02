from skimage import measure

from os import listdir, path

from imageio import imread

from math import pi

import numpy as np

from sklearn import preprocessing, model_selection

from scipy.stats import norm

import matplotlib.pyplot as pp

import seaborn as sb

# Load M Peg 7

mpeg7Path = './data/mpeg7'

mpeg7 = {}

for className in listdir(mpeg7Path):
    mpeg7.update(
        {
            className: {
                image: imread(path.join(mpeg7Path, className, image)) for image in sorted(listdir(path.join(mpeg7Path, className)))
            }
        }
    )

# Get morphological properties

mpeg7Properties = {}

mpeg7DataCache = []

mpeg7LabelsCache = []

for className, imagesDict in mpeg7.items():
    mpeg7Properties.update({className: {}})

    for imageName, imageMatrix in imagesDict.items():
        # Get labels

        imageLabeled = measure.label(imageMatrix, background=0)

        # Get properties

        for prop in measure.regionprops(imageLabeled, imageMatrix):
            mpeg7Properties[className].update({
                imageName: [
                    prop.area,
                    prop.eccentricity,
                    prop.mean_intensity,
                    prop.solidity,
                    (4 * pi * prop.area) /
                    prop.perimeter ** float(2)  # Circularity
                ]
            })

            mpeg7DataCache.append(mpeg7Properties[className][imageName])

            mpeg7LabelsCache.append(className)

mpeg7Data = np.vstack(mpeg7DataCache)

lp = preprocessing.LabelEncoder()

lp.fit(mpeg7LabelsCache)

mpeg7Labels = lp.transform(mpeg7LabelsCache)

mpeg7LabelsNames = lp.classes_

# Split dataset

x_train, x_test, y_train, y_test = model_selection.train_test_split(
    mpeg7Data,
    mpeg7Labels,
    test_size=0.25
)  # Data x_train / labels y_train

# Normal distribution
#
# www.scipy-lectures.org/intro/scipy/auto_examples/plot_normal_distribution.html
# 
# www.tutorialspoint.com/python/python_normal_distribution.htm
#
# docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.normal.html
#
# seaborn.pydata.org/generated/seaborn.distplot.html

np.random.seed(0)

mpeg7DataNormalized = preprocessing.minmax_scale(mpeg7Data)

mu, sigma = mpeg7DataNormalized.mean(), mpeg7DataNormalized.std()

# pp.figure(figsize=(8,4))
#
# count, bins, ignored = pp.hist(
#     np.random.normal(mu, sigma, 1000),
#     len(mpeg7DataNormalized),
#     density=True,
# )
#
# pp.plot(bins,
#     1 / (sigma * np.sqrt(2 * np.pi)) * \
#          np.exp(- (bins - mu)**2 / (2 * sigma**2)),
#     linewidth = 1,
#     color = 'r'
# )

sb.set()

ax = sb.distplot(np.random.normal(mu, sigma, 1000))

ax.set_title('Normal Distribution')

pp.show()
