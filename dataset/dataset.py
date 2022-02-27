# @Time    : 2022/2/23 16:21
# @Author  : ZYF
import os
from abc import ABCMeta

import pandas as pd

from database.influxdbtool import query_field_of_tags
from raw_time_series import RawTimeSeries
from utils.log import *


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
        for ts_name in ts_name_list:
            logging.info(f"Reading {ts_name} of {name}...")
            train = pd.read_csv(f'{path}/train/{ts_name}.txt', header=None, names=[f'dim_{i}' for i in range(1, 39)])
            train['label'] = 0
            test = pd.read_csv(f'{path}/test/{ts_name}.txt', header=None, names=[f'dim_{i}' for i in range(1, 39)])
            label_df = pd.read_csv(f'{path}/test_label/{ts_name}.txt', header=None, names=['label'])
            test['label'] = label_df['label']
            print(name, ts_name, len(train), len(test), len(label_df), len(train) + len(test))
            dataset.ts.append(RawTimeSeries.union(train=train, test=test, ds_name=name, ts_name=ts_name))
        return dataset

    @classmethod
    def load(cls, name):
        ts_name_list = query_field_of_tags({'dataset': name}, 'name')
        print(ts_name_list)
        dataset = Dataset(name)
        for ts_name in ts_name_list:
            dataset.ts.append(RawTimeSeries.load(ts_name))
        return dataset

    def save(self):
        for ts in self.ts:
            logging.info(f"Saving {ts.ts_name} of {self.name}...")
            ts.save()


if __name__ == '__main__':
    # smd = Dataset.fetch_SMD(
    #     '/Users/zhaoyunfeng/Desktop/实验室/智能运维/TSAD_Evaluator/data/SMD/OmniAnomaly-master/ServerMachineDataset')
    # smd.save()
    smd2 = Dataset.load('SMD')
    for ts in smd2.ts:
        print(ts.ts_name, len(ts.data))
