# @Time    : 2022/2/23 14:35
# @Author  : ZYF
import abc
from abc import ABCMeta
from datetime import datetime

import pandas as pd


class TimeSeries(metaclass=ABCMeta):
    def __init__(self, data: pd.DataFrame):
        super(TimeSeries, self).__init__()
        self.data = data.copy(deep=True)
        # 判断是否有时间索引
        if not isinstance(self.data.index, pd.DatetimeIndex):
            self.data = self.data.apply(pd.to_numeric)
            for ts_column in ['ts', 'time', 'timestamp']:
                if ts_column in self.data.columns:
                    self.data[ts_column] = self.data[ts_column].apply(pd.to_numeric)
                    ratio = 10 ** (max(0, len(str(int(self.data[ts_column][0]))) - 10))
                    timestamp = [ts / ratio for ts in self.data[ts_column]]
                    self.data.drop(columns=[ts_column], inplace=True)
                    break
            else:
                # 默认从0开始，一分钟一个点
                timestamp = [i * 60 for i in range(len(self.data))]
            timestamp = [ts if ts > (
                tmp := datetime.strptime('2022-01-01 00:00:00', '%Y-%m-%d %H:%M:%S').timestamp()) else ts + tmp
                         for ts in timestamp]
            self.data.index = pd.to_datetime(timestamp, unit='s')

    @abc.abstractmethod
    def gen_table_name(self):
        pass


if __name__ == '__main__':
    pass
