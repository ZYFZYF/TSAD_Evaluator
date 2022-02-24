# @Time    : 2022/2/23 14:35
# @Author  : ZYF
from abc import ABCMeta

import pandas


class TimeSeries(pandas.DataFrame, metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        super(TimeSeries, self).__init__(*args, **kwargs)
        if 'timestamp' not in self.columns:
            self['timestamp'] = list(range(len(self)))
        if 'label' not in self.columns:
            raise ValueError("should have label(0 for anomaly 1 for normal) column")
        self.sort_values(by=['timestamp'], inplace=True)

    def split(self, timestamp):
        return TimeSeries(self[self.timestamp < timestamp]), TimeSeries(
            self[self.timestamp >= timestamp])

    @classmethod
    def union(cls, train: pandas.DataFrame, test: pandas.DataFrame):
        return cls(pandas.concat([train, test]))


if __name__ == '__main__':
    df = TimeSeries(data={'timestamp': [1, 2, 3, 4, 5], 'value': [100, 200, 300, 400, 500], 'label': [0, 0, 1, 1, 1]})
    print(df)
    train, test = df.split(3)
    print(train)
    print(test)
    print(TimeSeries.union(train, test))
    tf = pandas.DataFrame(df)
    print(tf)
    df = TimeSeries(data={'value': [3, 4, 5, 6, 200, 300, 400, 500], 'label': [0, 0, 1, 0, 1, 0, 1, 1]})
    print(df)
    df = TimeSeries(data={'value': [3, 4, 5, 6, 200, 300, 400, 500]})
    print(df)
