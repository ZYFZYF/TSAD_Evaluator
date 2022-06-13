# @Time    : 2022/3/16 16:25
# @Author  : ZYF
import numpy as np

from algorithm.spot import biSPOT
from detector.detector import UnivariateDetector
from detector.fit import UnwantedFit
from detector.predict import StreamingPredict


class EVT(UnivariateDetector, UnwantedFit, StreamingPredict):
    def __init__(self):
        super(EVT, self).__init__()
        self.model = None

    def init(self, x: np.ndarray):
        self.model = biSPOT()
        self.model.initialize(init_data=x[:, 0])

    def predict(self, x: np.ndarray):
        result = self.model.detect(x[-1][0])
        return {'upper_threshold': result['upper_threshold'],
                'lower_threshold': result['lower_threshold'],
                'score': 1 if result['alarm'] else 0,
                'threshold': 0.5}
