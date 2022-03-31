# @Time    : 2022/3/30 21:43
# @Author  : ZYF
import numpy as np
import stumpy

from detector.detector import UnivariateDetector
from detector.fit import UnsupervisedFit
from detector.predict import OfflinePredict


class MatrixProfile(UnivariateDetector, UnsupervisedFit, OfflinePredict):

    def __init__(self, window_size=20):
        super(MatrixProfile, self).__init__()
        self.window_size = window_size
        self.train_data = []

    def fit(self, x: np.ndarray):
        self.train_data = x[:, 0].tolist()

    def predict(self, x: np.ndarray):
        return stumpy.stump(self.train_data + x[:, 0].tolist(), self.window_size)[len(self.train_data):, 0].tolist() + [
            0] * (self.window_size - 1)
