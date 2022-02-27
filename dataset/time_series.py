# @Time    : 2022/2/23 14:35
# @Author  : ZYF
import abc
from abc import ABCMeta

import pandas as pd


class TimeSeries(metaclass=ABCMeta):
    def __init__(self, data: pd.DataFrame):
        super(TimeSeries, self).__init__()
        self.data = data.copy(deep=True)
        # 判断是否有时间索引
        if not isinstance(self.data.index, pd.DatetimeIndex):
            for ts_column in ['ts', 'time', 'timestamp']:
                if ts_column in self.data.columns and self.data[ts_column].dtype == float:
                    ratio = 10 ** (max(0, 19 - len(str(int(self.data[ts_column][0])))))
                    x = self.data[ts_column][0]
                    print(str(int(x)), x, ratio)
                    timestamp = [ts * ratio for ts in self.data[ts_column]]
                    self.data.drop(columns=[ts_column])
                    break
            else:
                # 默认从0开始，一分钟一个点
                timestamp = [i * 60 * 1e9 for i in range(len(self.data))]
            self.data.index = pd.to_datetime(timestamp)

    @abc.abstractmethod
    def gen_table_name(self):
        pass


if __name__ == '__main__':
    pass
