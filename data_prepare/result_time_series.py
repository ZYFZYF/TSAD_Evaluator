# @Time    : 2022/3/6 10:46
# @Author  : ZYF
import pandas as pd

from data_prepare.time_series import TimeSeries
from database.influxdbtool import save_ts_data, save_meta_data
from utils.utils import get_meta_data


class ResultTimeSeries(TimeSeries):
    def __init__(self, data: pd.DataFrame, ds_name, ts_name, detector_name, eval_result):
        super(ResultTimeSeries, self).__init__(data)
        self.ds_name = ds_name
        self.ts_name = ts_name
        self.detector_name = detector_name
        for k, v in eval_result:
            self.__dict__[k] = v
        print(get_meta_data(self))

    def gen_table_name(self):
        return f'{self.detector_name}@{self.ds_name}@{self.ts_name}'

    def get_score(self):
        return

    def save(self):
        save_ts_data(data=self.data, table_name=self.gen_table_name())
        save_meta_data(tags={'name': self.gen_table_name()}, fields=get_meta_data(self), measurement='result_ts_meta')
