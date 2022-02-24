# @Time    : 2022/2/23 16:21
# @Author  : ZYF
from abc import ABCMeta


class Dataset(dict, metaclass=ABCMeta):
    def __init__(self, name):
        super().__init__()
        self._name = name
        self._time_series_list = list()


if __name__ == '__main__':
    pass
