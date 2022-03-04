# @Time    : 2022/3/2 14:29
# @Author  : ZYF
import abc
from enum import Enum


class FitMode(Enum):
    Supervised = "supervised"
    Unsupervised = "unsupervised"
    Unwanted = "unwanted"


class SupervisedFit(metaclass=abc.ABCMeta):
    fit_mode = FitMode.Supervised

    @abc.abstractmethod
    def fit(self, x, y):
        ...


class UnsupervisedFit(metaclass=abc.ABCMeta):
    fit_mode = FitMode.Unsupervised

    @abc.abstractmethod
    def fit(self, x):
        ...


class UnwantedFit(metaclass=abc.ABCMeta):
    fit_mode = FitMode.Unwanted
