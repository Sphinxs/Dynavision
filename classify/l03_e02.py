from sklearn import datasets, model_selection, metrics

from models.bayes import Bayes

import seaborn as sn

# from sklearn.naive_bayes import GaussianNB

import matplotlib.pyplot as pp

from matplotlib.colors import ListedColormap

# Load Iris dataset

iris = datasets.load_iris()

# Get sepal length and width

iris.data = iris.data[:, 0:1]

# Train / test split

x_train, x_test, y_train, y_test = model_selection.train_test_split(
    iris.data, iris.target,
    test_size=0.25)  # Data (x_train, y_train) | classes (x_test, y_test)

# Bayes

model = Bayes()  # GaussianNB()

model.fit(x_train, y_train)

predicted = model.predict(x_test)

# Confusion matrix

confusion = metrics.confusion_matrix(y_test, predicted[1])

print(f'\n\033[34mConfusion matrix\033[m : \n\n{confusion}')

pp.figure(figsize=(9, 3))

sn.heatmap(
    confusion, annot=True)

pp.show()

# Report

print(
    f'\n\033[34mClassification report\033[m : \n\n{metrics.classification_report(y_test, predicted)}',
    end='\n'
)

# Plot train / test

pp.scatter(
    x_train[:, 0],
    x_train[:, 0],
    c=y_train,
    cmap=ListedColormap(['r', 'g']),
    marker='.',
    edgecolors='none')

pp.scatter(
    x_test[:, 0],
    x_test[:, 0],
    c=y_test,
    cmap=ListedColormap(['y', 'b']),
    marker='1',
    edgecolors='none')

pp.show()