# @Time    : 2022/3/3 21:59
# @Author  : ZYF

class Detector:
    def __init__(self):
        self.name = None
        self.fit_mode = None
        self.predict_mode = None
        self.type = None


class UnivariateDetector(Detector):
    def __init__(self):
        super(UnivariateDetector, self).__init__()
        self.type = 'univariate'


class MultivariateDetector(Detector):
    def __init__(self):
        super(MultivariateDetector, self).__init__()
        self.type = 'multivariate'
