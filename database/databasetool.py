# @Time    : 2022/2/24 13:28
# @Author  : ZYF
import abc

from dataset.dataset import Dataset
from dataset.time_series import TimeSeries


class DatabaseTool(metaclass=abc.ABCMeta):
    @classmethod
    @abc.abstractmethod
    def load_time_series(cls, name):
        pass

    @classmethod
    @abc.abstractmethod
    def load_dataset(cls, name):
        pass

    @classmethod
    @abc.abstractmethod
    def save_time_series(cls, name, time_series: TimeSeries):
        pass

    @classmethod
    @abc.abstractmethod
    def save_dataset(cls, dataset: Dataset):
        pass
