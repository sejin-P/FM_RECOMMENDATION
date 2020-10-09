# import sys

# sys.path.extend(['/Users/parksejin/Line'])

import numpy as np

from criteo_conversion_logs.real.base_model import Model


class FM(Model):
    def set_weight(self):
        """
        initializing weight by uniform function range = 0~ sqrt(sqrt(len(feature counts)))
        :return:
        """
        np.random.seed(123)
        weight = np.random.uniform(0, 1 / np.math.sqrt(np.math.sqrt(len(self.one_hot_encoding[0]))),
                                   (len(self.one_hot_encoding[0])))
        self.weight = weight

    def phi(self, x: np.ndarray):
        """
        phi function - FM
        :param x:
        :return:
        """
        s = np.dot(x, self.weight)
        result = np.square(s)
        result -= np.sum(np.square(np.multiply(x, self.weight)), axis=-1)
        return result / 2

    def sub_gradient(self, x, y, lambd):
        s = np.dot(self.weight, x)
        difference = (s * x / 2) - np.multiply(self.weight, np.square(x))
        p = self.probability(x)
        return -((y / p) - (1 - y) / (1 - p)) * (np.exp(-self.phi(x)) / ((1 + np.exp(-self.phi(x))) ** 2)) * difference
