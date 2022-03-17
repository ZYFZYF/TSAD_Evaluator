# @Time    : 2022/3/7 10:12
# @Author  : ZYF
import abc


class Transform(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def transform(self, train: list[float], test: list[float]) -> (list[float], list[float]):
        ...


class TrivialTransform(Transform):
    def transform(self, train: list[float], test: list[float]) -> (list[float], list[float]):
        return train, test
