# @Time    : 2022/2/23 16:21
# @Author  : ZYF
import json
import os
import pickle
from abc import ABCMeta
from glob import glob

import pandas as pd
from tqdm import tqdm

from config import RAW_TIME_SERIES_MEASUREMENT, LABEL_COLUMN
from data_prepare.raw_time_series import RawTimeSeries
from database.influxdbtool import query_field_of_tags


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
    def fetch_KPI(cls, path: str, name='KPI'):
        TS_NAME_COLUMN = 'KPI ID'
        dataset = Dataset(name)
        # 初赛数据（已有论文均不评估预赛数据集，说是58条序列，实际上是29条，train和test分别是前半部分和后半部分
        # 复赛数据，分为train和test，但是都有异常，包括train
        all_train = pd.read_csv(f'{path}/Finals_dataset/phase2_train.csv')
        all_test = pd.read_hdf(f'{path}/Finals_dataset/phase2_ground_truth.hdf')
        all_test[TS_NAME_COLUMN] = [str(x) for x in all_test[TS_NAME_COLUMN]]
        ts_name_list = list(set(all_train[TS_NAME_COLUMN].tolist()))
        for ts_name in tqdm(iterable=sorted(ts_name_list), desc=f'Reading {name}'):
            train = all_train[all_train[TS_NAME_COLUMN] == ts_name].drop(columns=[TS_NAME_COLUMN])
            test = all_test[all_test[TS_NAME_COLUMN] == ts_name].drop(columns=[TS_NAME_COLUMN])
            dataset.ts.append(RawTimeSeries.union(train=train, test=test, ds_name=name, ts_name=ts_name))
        return dataset

    @classmethod
    def fetch_SKAB(cls, path: str, name='SKAB'):
        dataset = Dataset(name)
        for csv_file in tqdm(iterable=sorted(glob(f'{path}/data/*/*.csv')), desc=f'Reading {name}'):
            ts_name = '-'.join(csv_file.split('/')[-2:]).split('.')[0]
            if ts_name != 'anomaly-free-anomaly-free':
                data = pd.read_csv(csv_file, sep=';')
                dataset.ts.append(RawTimeSeries(data=data, ds_name=name, ts_name=ts_name))
        return dataset

    @classmethod
    def fetch_JumpStarter(cls, path: str, name='JumpStarter'):
        dataset = Dataset(name)
        for csv_file in tqdm(iterable=sorted(glob(f'{path}/dataset/Dataset[2|3]/train/*.csv')),
                             desc=f'Reading {name}'):
            ts_name = '-'.join(csv_file.split('/')[-3::2]).split('.')[0]
            train = pd.read_csv(csv_file, header=None, names=[f'dim_{i}' for i in range(1, 20)])
            train['label'] = 0
            test = pd.read_csv(csv_file.replace('train', 'test'), header=None,
                               names=[f'dim_{i}' for i in range(1, 20)])
            label_df = pd.read_csv(csv_file.replace('train', 'test_label'), header=None, names=['label'])
            test['label'] = label_df['label']
            dataset.ts.append(RawTimeSeries.union(train=train, test=test, ds_name=name, ts_name=ts_name))
        return dataset

    @classmethod
    def fetch_NAB(cls, path: str, name='NAB'):
        # TODO 这label是给人用的？
        dataset = Dataset(name)
        labels = {}
        for label_file in glob(f'{path}/labels/*/*.json') + glob(f'{path}/labels/*.json'):
            labels[label_file] = json.load(open(label_file, 'r'))
        for csv_file in glob(f'{path}/data/*/*.csv'):
            ts_name = '/'.join(csv_file.split('/')[-2:])
            for k, v in labels.items():
                if ts_name in v:
                    print(ts_name, k, v[ts_name])
        return dataset

    @classmethod
    def fetch_Industry(cls, path: str, name='Industry'):
        data = pickle.load(open(f'{path}/data/industry/industry_data_dict.pkl', 'rb'))
        dataset = Dataset(name)
        for ts_name, values in data.items():
            df = pd.DataFrame(data=values)
            df.rename(columns={'values': 'value', 'labels': LABEL_COLUMN}, inplace=True)
            print(ts_name, len(df))
            dataset.ts.append(RawTimeSeries(data=df, ds_name=name, ts_name=ts_name))
        return dataset

    @classmethod
    def fetch_UCR(cls, path: str, name='UCR'):
        data_dir = f'{path}/UCR_TimeSeriesAnomalyDatasets2021/FilesAreInHere/UCR_Anomaly_FullData'
        dataset = Dataset(name)

        for ts_file in tqdm(sorted(os.listdir(data_dir))):
            content_split = ts_file.split('.')[0].split('_')
            ts_name = '_'.join(content_split[:-3])
            train_data_len = int(content_split[-3])
            df = pd.read_csv(f'{data_dir}/{ts_file}', names=['value'])
            if len(df) == 1:
                df = pd.DataFrame(
                    data={'value': [float(val.strip()) for val in
                                    open(f'{data_dir}/{ts_file}', 'r').readlines()[0].split(' ') if
                                    len(val.strip()) > 0]})
            anomaly_start, anomaly_end = int(content_split[-2]), int(content_split[-1])
            df['label'] = [1 if anomaly_start <= i <= anomaly_end else 0 for i in range(1, len(df) + 1)]
            dataset.ts.append(RawTimeSeries(data=df, ds_name=name, ts_name=ts_name, train_data_len=train_data_len))
        return dataset

    @classmethod
    def fetch_TODS(cls, path: str, name='TODS'):
        data_dir = f'{path}/benchmark/synthetic/unidataset'
        dataset = Dataset(name)
        for ts_file in tqdm(sorted(os.listdir(data_dir))):
            ts_name = '.'.join(ts_file.split('.')[:-1]).replace('.', '_')
            df = pd.read_csv(f'{data_dir}/{ts_file}')
            dataset.ts.append(RawTimeSeries(data=df, ds_name=name, ts_name=ts_name))
        return dataset

    @classmethod
    def load(cls, name):
        ts_name_list = query_field_of_tags(measurement=RAW_TIME_SERIES_MEASUREMENT, tags={'dataset': name},
                                           field='name')
        dataset = Dataset(name)
        for ts_name in tqdm(iterable=ts_name_list, desc=f'Loading {name}'):
            dataset.ts.append(RawTimeSeries.load(ts_name))
        return dataset

    def save(self):
        for ts in tqdm(iterable=self.ts, desc=f'Saving {self.name}'):
            ts.save()


if __name__ == '__main__':
    # Dataset.fetch_SMD('../data/SMD/OmniAnomaly-master/ServerMachineDataset').save()
    # Dataset.fetch_SKAB('../data/SKAB/SKAB-master').save()
    # Dataset.fetch_JumpStarter('../data/JumpStarter/JumpStarter-main').save()
    # Dataset.fetch_NAB('../data/NAB/NAB-master').save()
    # Dataset.fetch_Industry('../data/Industry/ADSketch-main').save()
    # Dataset.fetch_Yahoo('../data/Yahoo/ydata-labeled-time-series-anomalies-v1_0').save()
    # Dataset.fetch_KPI('../data/KPI/KPI-Anomaly-Detection-master').save()
    # Dataset.fetch_UCR('../data/UCR/AnomalyDatasets_2021').save()
    Dataset.fetch_TODS('../data/TODS/tods-benchmark').save()
