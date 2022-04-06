# @Time    : 2022/4/6 22:47
# @Author  : ZYF
from typing import Callable

import numpy as np
from detector.detector import UnivariateDetector
from detector.fit import UnwantedFit
from detector.predict import TriggerPredict


class KSigma(UnwantedFit, UnivariateDetector, TriggerPredict):
    def __init__(self, train_data_len=30, test_data_len=10):
        super(KSigma, self).__init__()
        self.train_data_len = train_data_len
        self.test_data_len = test_data_len

    def predict(self, dataFetcher: Callable[[int], np.ndarray]):
        history_data = dataFetcher(self.train_data_len + self.test_data_len)
        train_data = history_data[:self.train_data_len][:, 0]
        test_data = history_data[self.train_data_len:][:, 0]
        train_mean = np.mean(train_data)
        train_std = np.std(train_data)
        fixed_std = max(train_std, max(0.01 * train_mean, 1e-3))
        return min(100, max([(i - train_mean) / fixed_std for i in test_data]))
