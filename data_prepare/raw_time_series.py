# @Time    : 2022/2/27 19:56
# @Author  : ZYF
import numpy as np
import pandas as pd

from config import DATASET_TEST_SPLIT, DELIMITER, RAW_TIME_SERIES_MEASUREMENT
from data_prepare.time_series import TimeSeries
from database.influxdbtool import save_ts_data, save_meta_data, load_ts_data, load_meta_data


class RawTimeSeries(TimeSeries):

    def __init__(self, data: pd.DataFrame, ds_name, ts_name, train_data_len=None):
        super(RawTimeSeries, self).__init__(data)
        self.ds_name = ds_name
        self.ts_name = ts_name
        if train_data_len is None:
            self.train_data_len = int(len(self.data)) * (1 - DATASET_TEST_SPLIT)
        else:
            self.train_data_len = train_data_len
        # 判断是否传入了label字段
        for label_column in ['label', 'anomaly', 'is_anomaly']:
            if label_column in self.data.columns:
                self.data.rename(columns={label_column: 'label'}, inplace=True)
                break
        else:
            self.data['label'] = 0
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
        return self.test_df.drop(columns=['label']).values, self.test_df['label'].values

    def get_train_data(self) -> (np.ndarray, np.ndarray):
        return self.train_df.drop(columns=['label']).values, self.train_df['label'].values

    def get_columns(self) -> list[str]:
        return self.data.columns

    def get_column_data(self, column_name):
        return RawTimeSeries(data=self.data[[column_name, 'label']], ds_name=self.ds_name,
                             ts_name=self.ts_name + '_' + column_name,
                             train_data_len=self.train_data_len)

    def is_univariate(self):
        return self.dim_num == 1

    def is_multivariate(self):
        return self.dim_num > 1
