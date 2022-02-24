# @Time    : 2022/2/23 16:21
# @Author  : ZYF
from abc import ABCMeta

from dataset.time_series import TimeSeries


class Dataset(dict, metaclass=ABCMeta):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def __setitem__(self, key: str, value: TimeSeries):
        if isinstance(key, str) and isinstance(value, TimeSeries):
            super(Dataset, self).__setitem__(key, value)
        else:
            raise ValueError("key should be str and value should be TimeSeries")


if __name__ == '__main__':
    dataset = Dataset("name")
    dataset['b'] = TimeSeries(data={'label': [1, 2, 3]})
    for x in dataset.items():
        print(x)
    print(dataset.__dict__)
    print(dir(dataset))
    dataset['a'] = 'b'
