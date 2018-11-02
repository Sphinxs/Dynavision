import numpy as np

from math import exp

def sigmoid(x):
    '''
        Calculate the Sigmoid curve for a scalar (activation function)

        Parameters
        ----------

        x:
            The input arrange

        Returns
        -------

        Returns the x of the given scalar

        Usage
        -----

        >>> sigmoid(5.0)

        Extra
        -----

        To calculate array's Sigmoid: numpy.exp()
    '''

    return 1.0 / (1.0 + exp(-x))

def lost():
    '''
        Evaluate the goodness of the predicted output (feedforward) using sum-of-squares error

        Parameters
        ----------

            None

        Returns
        -------

        None

        Usage
        -----

            >>> None

        Extra
        -----

        The calculus of feedforward is done by:

            for x in range(n): y - ỹ ** 2
    '''

    pass

class Neuro:
    '''
        Neural Network
    '''

    def __init__(self, x, y):
        '''
            Defines a Neural Network with two hidden layers

            Parameters
            ----------

            x:
                The input array

            y:
                The shape of the output object

            Returns
            -------

            None

            Usage
            -----

            >>> None
        '''

        self._input = x  # Input layer

        self._y = y  # Output layer

        self._output = np.ones(self._y.shape)  # Output shape

        self._weight1 = np.random.rand(self._input[0], 4)  # Layer one weight

        self._weight2 = np.random.rand(4, 1)  # Layer two weight

    def feedfoward(self):
        '''
            Calculate the predicted output (training)

            Parameters
            ----------

            None

            Returns
            -------

            None

            Usage
            -----

            >>> None

            Extra
            -----

            The calculus of feedforward is done by:

                W² (W¹ x + b¹) + b² 
        '''

        # Hidden layer, product of two arrays

        self._layer1 = sigmoid(np.dot(self._input, self._weight1))

        # Output layer, product of two arrays

        self._output = sigmoid(np.dot(self._layer1, self._weight2))

    def backpropagation(self):
        '''
            Update the weights and de biases
        '''

        pass
