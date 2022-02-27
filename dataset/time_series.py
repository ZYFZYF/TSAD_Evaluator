# @Time    : 2022/2/23 14:35
# @Author  : ZYF
from abc import ABCMeta

import pandas
import pandas as pd

from utils.config import CONFIG


class TimeSeries(pandas.DataFrame, metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        super(TimeSeries, self).__init__(*args, **kwargs)
        # 判断是否有时间索引
        if not isinstance(self.index, pandas.DatetimeIndex):
            for ts_column in ['ts', 'time', 'timestamp']:
                if ts_column in self.columns and self[ts_column].dtype == float:
                    ratio = 10 ** (max(0, 19 - len(str(int(self[ts_column][0])))))
                    x = self[ts_column][0]
                    print(str(int(x)), x, ratio)
                    timestamp = [ts * ratio for ts in self[ts_column]]
                    self.drop(columns=[ts_column])
                    break
            else:
                # 默认从0开始，一分钟一个点
                timestamp = [i * 60 * 1e9 for i in range(len(self))]
            self.index = pd.to_datetime(timestamp)
        # 判断是否传入了
        if 'label' not in self.columns:
            raise ValueError("should have label(0 for anomaly 1 for normal) column")

    def split(self):
        if hasattr(self, 'train_data_length'):
            print('has!!!')
            split_pos = self.train_data_length
        else:
            print('not has!!!')
            split_pos = int(len(self) * CONFIG.get('dataset').get('split_ratio'))

        return TimeSeries(self.iloc[:split_pos]), TimeSeries(self.iloc[split_pos:])

    @classmethod
    def union(cls, train: pandas.DataFrame, test: pandas.DataFrame):
        res = cls(pandas.concat([train, test]))
        res.train_data_length = len(train)
        return res


if __name__ == '__main__':
    df = TimeSeries(data={'timestamp': [1, 2, 3, 4, 5], 'value': [100, 200, 300, 400, 500], 'label': [0, 0, 1, 1, 1]})
    print(df)
    train, test = df.split()
    print(train)
    print(test)
    wf = TimeSeries.union(train, test)
    print(wf)
    print(wf.split())
    tf = pandas.DataFrame(df)
    print(tf)
    df = TimeSeries(data={'value': [3, 4, 5, 6, 200, 300, 400, 500], 'label': [0, 0, 1, 0, 1, 0, 1, 1]})
    print(df)
    df = TimeSeries(data={'value': [1], 'ts': [1645687819855.586], 'label': [0]})
    print(df)
    df = TimeSeries(data={'value': [3, 4, 5, 6, 200, 300, 400, 500]})
    print(df)
