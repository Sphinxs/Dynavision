def formatFigure(figure):
    '''
        Format figure 2D and 3D (gray, rgb or rgba)

        Parameter
        ---------

        figure:
            Image in format of array Numpy

        Return
        ------

        Figure formated
    '''

    from skimage import img_as_float

    figure = img_as_float(figure)

    if len(figure.shape) > 2:
        return segment(figure[:, :, 0])

    return figure


def segment(figure):
    '''
        Apply Histogram equalize and Otsu filter

        Parameter
        ---------

        figure:
            Image in format of array Numpy

        Return
        ------

        Segmented figure
    '''

    from skimage.filters import threshold_otsu

    from skimage.exposure import equalize_hist

    return figure > threshold_otsu(equalize_hist(figure))


def plot(axe, row, figures, test_labels):
    '''
        Plot figures

        Parameters
        ----------

        axe:
            Matplotlib subplot

        row:
            Subplot's row

        figures:
            List of images in format of array Numpy

        test_labels:
            Test labels

        Return
        ------

        Figures ploted
    '''

    from sklearn import metrics

    import seaborn

    count = 0

    for key, value in figures:
        confusionMatrix = metrics.confusion_matrix(test_labels, value)

        seaborn.heatmap(
            confusionMatrix,
            annot=True,
            ax=axe[row, count],
            cbar=False
        )

        axe[row, count].set_title(key)

        axe[row, count].axis('off')

        count += 1

        print(f'\n\033[31m{key}\033[m\n\n{metrics.classification_report(test_labels, value)}\n',)


def morphologic(folder, simple=False):
    '''
        Morphological properties

        Parameters
        ----------

        folder:
            Dict of images in format of array Numpy

        simple:
            Simple reading

        Return
        ------

        Morphological properties
    '''

    from skimage import measure

    from math import pi

    properties = []

    labels = []

    if simple:
        for imageName, imageData in folder.items():
            # Labels
            imageLabeled = measure.label(imageData, background=0)

            # Properties
            for prop in measure.regionprops(imageLabeled, imageData, coordinates='rc'):
                properties.append([
                    prop.area,
                    prop.eccentricity,
                    prop.mean_intensity,
                    prop.solidity,
                    # (4 * pi * prop.area) / prop.perimeter ** 2.0  # Circularity
                ])

                labels.append(imageName)
    else:
        for className, imagesDict in folder.items():
            for imageName, imageData in imagesDict.items():
                # Labels
                imageLabeled = measure.label(imageData, background=0)

                # Properties
                for prop in measure.regionprops(imageLabeled, imageData, coordinates='rc'):
                    properties.append([
                        prop.area,
                        prop.eccentricity,
                        prop.mean_intensity,
                        prop.solidity,
                        # (4 * pi * prop.area) / prop.perimeter ** 2.0  # Circularity
                    ])

                    labels.append(className)

    return [properties, labels]


def models(train_data, train_labels, test_data):
    '''
        Classify objects

        Parameters
        ----------

        train_data, train_labels:
            Train data and train labels

        test_data:
            Test data

        Return
        ------

        Prediction by multiple algorithms
    '''

    from sklearn.gaussian_process import GaussianProcessClassifier

    from sklearn.gaussian_process.kernels import RBF

    from sklearn.neighbors import KNeighborsClassifier

    from sklearn.naive_bayes import GaussianNB

    from sklearn.tree import DecisionTreeClassifier

    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

    from sklearn import svm

    from sklearn.neural_network import MLPClassifier

    prediction = {}

    # Gaussian Process Classifier

    prediction.update({
        'Gaussian Process':
            GaussianProcessClassifier(1.0 * RBF(1.0))
                .fit(train_data, train_labels)
                .predict(test_data)
    })

    # K-Nearest Neighbors

    prediction.update({
        'K-Nearest Neighbors':
            KNeighborsClassifier(n_neighbors=4)
                .fit(train_data, train_labels)
                .predict(test_data)
    })

    # Gaussian Naive Bayes

    prediction.update({
        'Gaussian Naive Bayes':
            GaussianNB()
                .fit(train_data, train_labels)
                .predict(test_data)
    })

    # Decision Tree

    prediction.update({
        'Decision Tree':
            DecisionTreeClassifier()
                .fit(train_data, train_labels)
                .predict(test_data)
    })

    # Random Forest

    prediction.update({
        'Random Forest':
            RandomForestClassifier(n_estimators=20)
                .fit(train_data, train_labels)
                .predict(test_data)
    })

    # Ada Boost

    prediction.update({
        'Ada Boost':
            AdaBoostClassifier()
                .fit(train_data, train_labels)
                .predict(test_data)
    })

    # Suport Vector Machine

    prediction.update({
        'Suport Vector Machine':
            svm.SVC(gamma='scale')
                .fit(train_data, train_labels)
                .predict(test_data)
    })

    # Multi-layer Perceptron

    prediction.update({
        'Multi-layer Perceptron':
            MLPClassifier(alpha=1)
                .fit(train_data, train_labels)
                .predict(test_data)
    })

    return prediction
