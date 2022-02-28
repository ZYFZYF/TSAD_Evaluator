# @Time    : 2022/2/23 16:21
# @Author  : ZYF
import os
from abc import ABCMeta
from glob import glob

import pandas as pd
from tqdm import tqdm

from database.influxdbtool import query_field_of_tags
from raw_time_series import RawTimeSeries


class Dataset(metaclass=ABCMeta):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.ts = []

    # TODO 支持自动下载并解压
    @classmethod
    def fetch_SMD(cls, path: str, name='SMD'):
        ts_name_list = [ts_name.split('.')[0] for ts_name in os.listdir(f'{path}/train')]
        dataset = Dataset(name)
        for ts_name in tqdm(iterable=sorted(ts_name_list), desc=f'Reading {name}'):
            train = pd.read_csv(f'{path}/train/{ts_name}.txt', header=None, names=[f'dim_{i}' for i in range(1, 39)])
            train['label'] = 0
            test = pd.read_csv(f'{path}/test/{ts_name}.txt', header=None, names=[f'dim_{i}' for i in range(1, 39)])
            label_df = pd.read_csv(f'{path}/test_label/{ts_name}.txt', header=None, names=['label'])
            test['label'] = label_df['label']
            dataset.ts.append(RawTimeSeries.union(train=train, test=test, ds_name=name, ts_name=ts_name))
        return dataset

    @classmethod
    def fetch_Yahoo(cls, path: str, name='Yahoo'):
        dataset = Dataset(name)
        for csv_file in tqdm(iterable=sorted(glob(f'{path}/*/*.csv')), desc=f'Reading {name}'):
            if (ts_name := os.path.splitext(os.path.basename(csv_file))[0]) not in ['A3Benchmark_all',
                                                                                    'A4Benchmark_all']:
                data = pd.read_csv(csv_file)
                data.rename(columns={'is_anomaly': 'label',
                                     'anomaly': 'label',
                                     'timestamps': 'timestamp'}, inplace=True)
                data = data[['timestamp', 'value', 'label']]
                dataset.ts.append(RawTimeSeries(data=data, ds_name=name, ts_name=ts_name))
        return dataset

    @classmethod
    def load(cls, name):
        ts_name_list = query_field_of_tags({'dataset': name}, 'name')
        dataset = Dataset(name)
        for ts_name in tqdm(iterable=ts_name_list, desc=f'Loading {name}'):
            dataset.ts.append(RawTimeSeries.load(ts_name))
        return dataset

    def save(self):
        for ts in tqdm(iterable=self.ts, desc=f'Saving {self.name}'):
            ts.save()


if __name__ == '__main__':
    # smd = Dataset.fetch_SMD(
    #     '/Users/zhaoyunfeng/Desktop/实验室/智能运维/TSAD_Evaluator/data/SMD/OmniAnomaly-master/ServerMachineDataset')
    # smd.save()
    # for ts in smd2.ts:
    #     print(ts.ts_name, len(ts.data))
    yahoo = Dataset.fetch_Yahoo(
        '/Users/zhaoyunfeng/Desktop/实验室/智能运维/TSAD_Evaluator/data/Yahoo/ydata-labeled-time-series-anomalies-v1_0')
    yahoo.save()
