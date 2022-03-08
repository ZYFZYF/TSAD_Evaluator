# @Time    : 2022/3/2 14:29
# @Author  : ZYF
import abc

import numpy as np

from detector.detector import Detector


class FitMode:
    Supervised = "supervised"
    Unsupervised = "unsupervised"
    Unwanted = "unwanted"


class SupervisedFit(Detector, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()
        self.fit_mode = FitMode.Supervised

    @abc.abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray):
        ...


class UnsupervisedFit(Detector, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()
        self.fit_mode = FitMode.Unsupervised

    @abc.abstractmethod
    def fit(self, x: np.ndarray):
        ...


class UnwantedFit(Detector, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()
        self.fit_mode = FitMode.Unwanted
