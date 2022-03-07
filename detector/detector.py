# @Time    : 2022/3/3 21:59
# @Author  : ZYF
from database.influxdbtool import save_meta_data


class Detector:
    def __init__(self):
        self.name = None
        self.fit_mode = None
        self.predict_mode = None
        self.type = None

    def save(self, algo_name):
        save_meta_data(measurement='detector_meta', tags={'name': algo_name}, fields={'fit_mode': self.fit_mode,
                                                                                      'predict_mode': self.predict_mode,
                                                                                      'type': self.type})


class UnivariateDetector(Detector):
    def __init__(self):
        super(UnivariateDetector, self).__init__()
        self.type = 'univariate'


class MultivariateDetector(Detector):
    def __init__(self):
        super(MultivariateDetector, self).__init__()
        self.type = 'multivariate'
