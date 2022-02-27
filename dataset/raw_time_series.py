# @Time    : 2022/2/27 19:56
# @Author  : ZYF
import pandas as pd

from database.influxdbtool import save_ts_data, save_meta_data, load_ts_data, load_meta_data
from time_series import TimeSeries
from utils.config import CONFIG
from utils.log import *


class RawTimeSeries(TimeSeries):

    def __init__(self, data, ds_name, ts_name, train_data_len=None):
        super(RawTimeSeries, self).__init__(data)
        self.ds_name = ds_name
        self.ts_name = ts_name
        if train_data_len is None:
            self.train_data_len = int(len(self.data) * CONFIG.get('dataset').get('split_ratio'))
        else:
            self.train_data_len = train_data_len
        # 判断是否传入了label字段
        if 'label' not in self.data.columns:
            raise ValueError("should have label(0 for anomaly 1 for normal) column")

    def split(self):
        return TimeSeries(self.data.iloc[:self.train_data_len]), TimeSeries(self.data.iloc[self.train_data_len:])

    @classmethod
    def union(cls, train: pd.DataFrame, test: pd.DataFrame, ds_name, ts_name):
        res = cls(pd.concat([train, test]), ds_name, ts_name)
        res.train_data_length = len(train)
        return res

    def gen_table_name(self):
        return self.ds_name + '@' + self.ts_name

    def save(self):
        save_ts_data(data=self.data, table_name=self.gen_table_name())
        save_meta_data(tags={'name': self.gen_table_name()}, fields={'dataset': self.ds_name,
                                                                     'ds_name': self.ds_name,
                                                                     'ts_name': self.ts_name,
                                                                     'train_data_len': self.train_data_len})

    # TODO 尝试把这块儿写的漂亮一点，现在这个键值对儿重复了好多次
    @classmethod
    def load(cls, name):
        logging.info(f"Loading data of {name}")
        data = load_ts_data(name)
        logging.info(f"Loading meta of {name}")
        meta = load_meta_data({'name': name})
        return cls(data=data, ds_name=meta['ds_name'], ts_name=meta['ts_name'], train_data_len=meta['train_data_len'])
