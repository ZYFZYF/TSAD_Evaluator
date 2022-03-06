# @Time    : 2022/3/2 14:32
# @Author  : ZYF
import abc
from enum import Enum

import numpy as np


class PredictMode(Enum):
    Offline = "offline"
    Stream = "stream"
    Trigger = "trigger"


class OfflinePredict(metaclass=abc.ABCMeta):
    predict_mode = PredictMode.Offline

    @abc.abstractmethod
    def predict(self, x: np.ndarray):
        ...


class StreamingPredict(metaclass=abc.ABCMeta):
    predict_mode = PredictMode.Stream

    @abc.abstractmethod
    def predict(self, x):
        ...


class TriggerPredict(metaclass=abc.ABCMeta):
    predict_mode = PredictMode.Trigger

    @abc.abstractmethod
    def predict(self, x):
        ...
