# @Time    : 2022/4/6 19:36
# @Author  : ZYF
import random
from typing import Callable

import numpy as np

from detector.detector import UnivariateDetector
from detector.fit import UnwantedFit
from detector.predict import TriggerPredict


class RandomTrigger(UnwantedFit, UnivariateDetector, TriggerPredict):
    def predict(self, dataFetcher: Callable[[int], np.ndarray]):
        # print(f'获取过去10个点的数据', dataFetcher(10))
        return random.random()
