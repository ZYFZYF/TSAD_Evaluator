# @Time    : 2022/2/27 19:56
# @Author  : ZYF
import random

import numpy as np
import pandas as pd

from config import DATASET_TEST_SPLIT, DELIMITER, RAW_TIME_SERIES_MEASUREMENT, LABEL_COLUMN, LABEL_COLUMN_NAMES
from data_prepare.time_series import TimeSeries
from database.influxdbtool import save_ts_data, save_meta_data, load_ts_data, load_meta_data


class RawTimeSeries(TimeSeries):

    def __init__(self, data: pd.DataFrame, ds_name, ts_name, train_data_len=None):
        super(RawTimeSeries, self).__init__(data)
        self.ds_name = ds_name
        self.ts_name = ts_name
        if train_data_len is None:
            self.train_data_len = int(len(self.data) * (1 - DATASET_TEST_SPLIT))
        else:
            self.train_data_len = train_data_len
        # 判断是否传入了label字段
        for label_column in LABEL_COLUMN_NAMES:
            if label_column in self.data.columns:
                self.data.rename(columns={label_column: LABEL_COLUMN}, inplace=True)
                break
        else:
            self.data[LABEL_COLUMN] = 0
        self.dim_num = len(self.data.columns) - 1
        self.train_df = self.data.iloc[:self.train_data_len]
        self.test_df = self.data.iloc[self.train_data_len:]

    def split(self):
        return TimeSeries(self.data.iloc[:self.train_data_len]), TimeSeries(self.data.iloc[self.train_data_len:])

    @classmethod
    def union(cls, train: pd.DataFrame, test: pd.DataFrame, ds_name, ts_name):
        res = cls(pd.concat([train, test]), ds_name, ts_name)
        res.train_data_length = len(train)
        return res

    def gen_table_name(self):
        return self.ds_name + DELIMITER + self.ts_name

    def save(self):
        save_ts_data(data=self.data, table_name=self.gen_table_name())
        save_meta_data(tags={'name': self.gen_table_name()}, fields={'dataset': self.ds_name,
                                                                     'ds_name': self.ds_name,
                                                                     'ts_name': self.ts_name,
                                                                     'train_data_len': self.train_data_len,
                                                                     'dim_num': self.dim_num},
                       measurement=RAW_TIME_SERIES_MEASUREMENT)

    # TODO 尝试把这块儿写的漂亮一点，现在这个键值对儿重复了好多次
    @classmethod
    def load(cls, name):
        data = load_ts_data(name)
        meta = load_meta_data(measurement=RAW_TIME_SERIES_MEASUREMENT, tags={'name': name})
        return cls(data=data, ds_name=meta['ds_name'], ts_name=meta['ts_name'], train_data_len=meta['train_data_len'])

    def get_test_data(self) -> (np.ndarray, np.ndarray):
        return self.test_df.drop(columns=[LABEL_COLUMN]).values, self.test_df[LABEL_COLUMN].values

    def get_train_data(self) -> (np.ndarray, np.ndarray):
        return self.train_df.drop(columns=[LABEL_COLUMN]).values, self.train_df[LABEL_COLUMN].values

    def get_columns(self) -> list[str]:
        return [col for col in self.data.columns if col != LABEL_COLUMN]

    def get_column_data(self, column_name):
        return RawTimeSeries(data=self.data[[column_name, LABEL_COLUMN]], ds_name=self.ds_name,
                             ts_name=self.ts_name + '_' + column_name,
                             train_data_len=self.train_data_len)

    def is_univariate(self):
        return self.dim_num == 1

    def is_multivariate(self):
        return self.dim_num > 1

    # TODO 保证单个序列每次选出来的区间一致
    def select_detect_range(self, window_size: int, anomaly_ratio: float, is_disjoint: bool) -> list[(int, int, int)]:
        random.seed(hash(self.ds_name))
        test_label = self.test_df[LABEL_COLUMN].values
        candidate = set(list(range(0, len(test_label) - window_size)))

        def remove(k):
            if is_disjoint:
                for n in range(k - window_size, k + window_size + 1):
                    candidate.discard(n)
            else:
                candidate.discard(k)

        detect_range_list = []
        i, j = 0, 0
        while i + window_size < len(test_label):
            if test_label[i] == 1:
                le = i - random.randint(1, window_size // 2)
                detect_range_list.append((le, le + window_size - 1, 1))
                remove(le)
                for j in range(i - window_size, i + 1):
                    candidate.discard(j)
                while i + window_size < len(test_label) and test_label[i + 1] == 1:
                    i += 1
                    candidate.discard(i)
            i += 1
        anomaly_range_cnt = len(detect_range_list)
        for i in range(int(anomaly_range_cnt * (1.0 - anomaly_ratio) / anomaly_ratio)):
            le = random.sample(candidate, 1)[0]
            remove(le)
            if len(candidate) == 0:
                break
            detect_range_list.append((le, le + window_size - 1, 0))
        return sorted(detect_range_list, key=lambda x: x[0])

    def get_test_timestamp(self):
        return self.test_df.index.values.astype(np.int64)


if __name__ == '__main__':
    ts = RawTimeSeries.load('Yahoo@A3Benchmark-TS15')
    print(ts.select_detect_range(10, 0.1, True))
    print(ts.select_detect_range(10, 0.05, False))
    print(ts.get_test_timestamp())
