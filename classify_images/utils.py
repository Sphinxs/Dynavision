def plot(ax, row, figs, y_test):
    '''
        Plot figures in a row

        Parameters
        ----------

        ax:
            Subplot

        row:
            Row on subplot

        figs:
            List of tuples (name, data)

        y_test:
            Test labels
        
        Return
        ------

        Figures ploted in a row of a subplot
    '''

    from sklearn import metrics

    import seaborn as sn

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
            f'\n{k} - \033[31mClassification Report\033[m:\n\n{metrics.classification_report(y_test, v)}\n',
        )

    return cache


def printFormated(obj, indent=3, sort_keys=False):
    '''
        Formated dict

        Parameters
        ----------

        obj:
            Object to format

        indent:
            Indentation size
        
        sort_keys:
            Dict keys
        
        Return
        ------

        None
    '''
    
    from json import dumps

    print('\n', dumps(
        obj,
        sort_keys=sort_keys,
        indent=indent,
        separators=('\n', ' : ')
    ), '\n')


def morphologic(folder, simple=False):
    '''
        Morphological properties from images

        Parameters
        ----------

        folder:
            Images data
        
        simple:
            Images from one folder
        
        Return
        ------

        Morphological properties
    '''

    from skimage import measure

    from math import pi

    properties = {}
    
    propertiesCache = []
    
    labelsCache = []

    if simple:
        for imageName, imageData in folder.items():
            # Remove alpha

            # imageData[:, :, :3]

            # convert name.png -background black -alpha remove -alpha off name.png

            # Get labels
            imageLabeled = measure.label(imageData, background=0)
            
            # Get properties
            for prop in measure.regionprops(imageLabeled, imageData, coordinates='rc'):
                properties.update({
                    imageName: [
                        prop.area,
                        prop.eccentricity,
                        prop.mean_intensity,
                        prop.solidity,
                        # (4 * pi * prop.area) / prop.perimeter ** 2.0  # Circularity
                    ]
                })

                propertiesCache.append(properties[imageName])

                labelsCache.append(imageName)
    else:
        for className, imagesDict in folder.items():
            properties.update({className: {}})

            for imageName, imageData in imagesDict.items():
                # Get labels
                imageLabeled = measure.label(imageData, background=0)

                # Get properties
                for prop in measure.regionprops(imageLabeled, imageData, coordinates='rc'):
                    properties[className].update({
                        imageName: [
                            prop.area,
                            prop.eccentricity,
                            prop.mean_intensity,
                            prop.solidity,
                            # (4 * pi * prop.area) / prop.perimeter ** 2.0  # Circularity
                        ]
                    })

                    propertiesCache.append(properties[className][imageName])

                    labelsCache.append(className)
    
    return [properties, propertiesCache, labelsCache]


def models(x_train, y_train, x_test):
    '''
        Classify data using Sklearn algorithms

        Parameters
        ----------

        x_train, y_train:
            Train data and train labels

        x_test:
            Test data

        Return
        ------

        Predicted object from various algorithms
    '''

    from sklearn.neighbors import KNeighborsClassifier

    from sklearn.naive_bayes import GaussianNB

    from sklearn import svm

    from sklearn.tree import DecisionTreeClassifier

    from sklearn.ensemble import RandomForestClassifier

    from sklearn.gaussian_process import GaussianProcessClassifier

    from sklearn.gaussian_process.kernels import RBF

    predicted = {}

    # Gaussian Process Classifier

    gaussian = GaussianProcessClassifier(1.0 * RBF(1.0))

    gaussian.fit(x_train, y_train)

    predicted.update({
        'Gaussian Process Classifier': gaussian.predict(x_test)
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
        'Gaussian Naive Bayes': bayes.predict(x_test)
    })

    # Decision Tree

    tree = DecisionTreeClassifier()

    tree.fit(x_train, y_train)

    predicted.update({
        'Decision Tree Classifier': tree.predict(x_test)
    })

    # Random Forest

    random = RandomForestClassifier(n_estimators=20)

    random.fit(x_train, y_train)

    predicted.update({
        'Random Forest Classifier': random.predict(x_test)
    })

    # Svm

    svm = svm.SVC(gamma='scale')  # 1 / (n_features * X.std())

    svm.fit(x_train, y_train)

    predicted.update({
        'Suport Vector Machine': svm.predict(x_test)
    })

    return predicted
