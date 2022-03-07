# @Time    : 2022/3/7 10:27
# @Author  : ZYF
import abc


class Threshold(abc.ABCMeta):
    @abc.abstractmethod
    def threshold(self, train: list[list[float]], test: list[list[float]]) -> list[float]:
        ...


class MaxTrainThreshold(Threshold):
    def threshold(self, train: list[list[float]], test: list[list[float]]):
        th = max(train)
        return [th for _ in range(len(test))]
