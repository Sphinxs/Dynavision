from skimage import measure

from os import listdir, path

from imageio import imread

from math import pi

import numpy as np

from sklearn import preprocessing, model_selection, metrics

from scipy.stats import norm

import matplotlib.pyplot as pp

import seaborn as sn

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn import svm

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

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

        for prop in measure.regionprops(imageLabeled, imageMatrix, coordinates='rc'):
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

x_train, x_test, y_train, y_test = model_selection.train_test_split( # Cross_val_score
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

sn.set()

ax = sn.distplot(np.random.normal(mu, sigma, 1000))

ax.set_title('Normal Distribution')

# pp.show()

# Classify using KNN, Naive Bayes (Gaussian), Decision Tree, Random Forest and SVM

predicted = {}

# Knn

knn = KNeighborsClassifier(n_neighbors=8)

knn.fit(x_train, y_train)

predicted.update({
    'K-Nearest Neighbors': knn.predict(x_test) # Y test
})

# Naive Bayes

bayes = GaussianNB()

bayes.fit(x_train, y_train)

predicted.update({
    'Naive Bayes': bayes.predict(x_test) # Y Test
})

# Decision Tree

tree = DecisionTreeClassifier(max_depth=5)

tree.fit(x_train, y_train)

predicted.update({
    'Decision Tree': tree.predict(x_test) # Y Test
})

# Random Forest

random = RandomForestClassifier()

random.fit(x_train, y_train)

predicted.update({
    'Random Forest': random.predict(x_test) # Y
})

# Svm

svm = svm.SVC()

svm.fit(x_train, y_train)

predicted.update({
    'Suport Vector Machine': svm.predict(x_test) # Y Test
})

# Get metrics and plot confusion matrix

fig, ax = pp.subplots(1, len(predicted), figsize=(len(predicted) * 2, 3))

predictedMetrics = {}

counter = 0

for k, v in predicted.items():
    confusion = metrics.confusion_matrix(y_test, v)
    
    sn.heatmap(confusion, annot=True, ax=ax[counter])

    ax[counter].set_title(k)
    
    # print(
    #    f'\n{k} - \033[34mReport\033[m:\n\n{metrics.classification_report(y_test, v)}', end='\n'
    # )

    predictedMetrics.update({
        k: {
            'accuracy_score': metrics.accuracy_score(y_test, v),
            'jaccard_similarity_score': metrics.jaccard_similarity_score(y_test, v)
        }
    })

    counter += 1

pp.show()
