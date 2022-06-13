# @Time    : 2022/3/2 14:32
# @Author  : ZYF
import abc

import numpy as np

from detector.detector import Detector
from typing import Callable


class PredictMode:
    Offline = "offline"
    Stream = "stream"
    Trigger = "trigger"


class OfflinePredict(Detector, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()
        self.predict_mode = PredictMode.Offline

    @abc.abstractmethod
    def predict(self, x: np.ndarray):
        ...


class StreamingPredict(Detector, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()
        self.predict_mode = PredictMode.Stream

    @abc.abstractmethod
    def init(self, x: np.ndarray):
        ...

    @abc.abstractmethod
    def predict(self, x: np.ndarray):
        ...


class TriggerPredict(Detector, metaclass=abc.ABCMeta):

    def __init__(self):
        super().__init__()
        self.predict_mode = PredictMode.Trigger

    @abc.abstractmethod
    def predict(self, dataFetcher: Callable[[int], np.ndarray]):
        ...
