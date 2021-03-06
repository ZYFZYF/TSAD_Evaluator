# @Time    : 2022/3/7 10:17
# @Author  : ZYF
import abc


class Aggregate(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def aggregate(self, train: list[list[float]], test: list[list[float]]) -> (list[float], list[float]):
        ...


class MaxAggregate(Aggregate):

    def aggregate(self, train: list[list[float]], test: list[list[float]]) -> (list[float], list[float]):
        return [max(x) for x in train], [max(x) for x in test]


class MeanAggregate(Aggregate):

    def aggregate(self, train: list[list[float]], test: list[list[float]]) -> (list[float], list[float]):
        return [sum(x) / len(x) for x in train], [sum(x) / len(x) for x in test]
