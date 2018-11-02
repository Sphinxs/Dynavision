from scipy import spatial

__all__ = ['Knn']

class Knn:
    def fit(self, x_train, y_train):
        '''
            Fit KNN model

            Parameters
            ----------

            x_train:
                Bi-dimensional Numpy array. Train data.

            y_train:
                Numpy array. Classes from train data.

            Usage
            -----

            >>> Knn().fit(numpy.array([
                [1.0, 1.5],
                [2.0, 2.5]
            ]))

            Return
            ------

            Prediction function
        '''

        self.x_train = x_train

        self.y_train = y_train

        def predict(x_test):
            self.x_test = x_test

            # Euclidian distance between each test object from data
            # and train object

            dist = spatial.distance.cdist(
                x_test, self.x_train, 'euclidean')  # Numpy array

            # Get indices from min distance

            min_dist = dist.argmin(axis=1)

            # Get classes from nearest train objects

            y_pred = self.y_train[min_dist]

            return y_pred

        return predict
