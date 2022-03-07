# @Time    : 2022/3/6 09:32
# @Author  : ZYF
from random import random

import numpy as np

from detector.detector import UnivariateDetector
from detector.fit import SupervisedFit
from detector.predict import OfflinePredict


class RandomDetector(SupervisedFit, UnivariateDetector, OfflinePredict):

    def fit(self, x: np.ndarray, y: np.ndarray):
        return [random() for _ in range(x.shape[0])]

    def predict(self, x: np.ndarray):
        return [random() for _ in range(x.shape[0])]
