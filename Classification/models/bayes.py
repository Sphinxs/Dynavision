import numpy as np

__all__ = ['Bayes']


class Bayes:
    '''
		Naive Bayes

		Reference
        ---------

        [1] CShape analisys and classification : theory and practice. Crc Press, 2000.
	'''

    def dist_normal(self, h, media, deviation):
        '''
			Normal distribution

			Parameters
			----------

			Usage
			-----

			Return
			------

		'''

        numerator = (1. / np.sqrt(2 * np.pi * deviation**2))

        denominator = np.exp(-((h - media)**2.) / (2 * deviation**2))

        return numerator * denominator

    def fit(self, x_train, y_train):
        '''
			Fit Bayes model

			Parameters
			----------

			Usage
			-----

			Return
			------

		'''

        # Class 0 and 1

        class_0 = x_train[y_train == 0]

        class_1 = x_train[y_train == 1]

        # Amount of samples in each class

        amount_class_0 = class_0.shape[0]

        amount_class_1 = class_1.shape[0]

        # Total of samples

        amount_total = amount_class_0 + amount_class_1

        # Mean and default deviation

        self.param_c0 = [np.mean(class_0, 0), np.std(class_0, 0)]

        self.param_c1 = [np.mean(class_1, 0), np.std(class_1, 0)]

        # Observed variable

        self.h = np.linspace(
            np.min([class_0.min(), class_1.min()]) - 1.,
            np.max([class_0.max(), class_1.max()]) + 1., 200)

        # Conditional density h knowing that the object belongs to f(h|C0) and f(h|C1)

        self.fh_c0 = self.dist_normal(self.h, self.param_c0[0],
                                      self.param_c0[1])

        self.fh_c1 = self.dist_normal(self.h, self.param_c1[0],
                                      self.param_c1[1])

        # Probability of h belongs to class c0 or c1 P(C1) and P(C2)

        self.p_c0 = float(amount_class_0) / amount_total

        self.p_c1 = float(amount_class_1) / amount_total

        # Weighted conditional comparison density function h P(C1)f(h|C1) and P(C2)f(h|C2)

        self.p_c0_fh_c0 = self.p_c0 * self.fh_c0

        self.p_c1_fh_c1 = self.p_c1 * self.fh_c1

    def predict(self, x_test):
        '''
			Predict objects not apressented to the model

			Parameters
			----------

			Usage
			-----

			Return
			------

		'''

        pred_i = np.zeros(x_test.shape[0])

        if self.p_c0 > self.p_c1:
            pred_i[:] = 0
        else:
            pred_i[:] = 1

        # Bayes II
        # --------
        # Classify new objects based on known characteristics.

        pred_ii = np.zeros(x_test.shape[0])

        # p(C0) f(h | C0) : Probability of a object pertence to the C0 after mensure the H

        p_c0_fh_c0_ = self.p_c0 * self.dist_normal(
            x_test[:, 0], self.param_c0[0], self.param_c0[1])

        # p(C1) f(h | C1) : Probability of a object pertence to the C1 after mensure the H

        p_c1_fh_c1_ = self.p_c1 * self.dist_normal(
            x_test[:, 0], self.param_c1[0], self.param_c1[1])

        # p(C0)f(h | 0) > P(C1)f(h | C1) : Class C0

        pred_ii[p_c0_fh_c0_ > p_c1_fh_c1_] = 0

        # P(C0)f(h | C0) <=P (C1)f(h | C1) : Class C1

        pred_ii[p_c0_fh_c0_ <= p_c1_fh_c1_] = 1

        return pred_i, pred_ii

    def decision(self):
        '''
		Decision region

		Parameters
		----------

		Usage
		-----

		Return
		------

	'''

        l_h = self.fh_c1 / self.fh_c0

        # Decision limear

        t = float(self.p_c0) / float(self.p_c1)

        # Likelihood values higher then T : Class C1

        h_c0 = self.h[l_h >= t]

        # Likelihood values higher then T : Class C2

        h_c1 = self.h[l_h < t]

        # Decision region

        self.regiao_decisao = (h_c0.min() + h_c1.max()) / 2.

        return t
