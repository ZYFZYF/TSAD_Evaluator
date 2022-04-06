# @Time    : 2022/3/3 21:51
# @Author  : ZYF
import logging
from typing import Union

import numpy as np
import pandas as pd
from tqdm import tqdm
from aggregate.aggregate import Aggregate, MaxAggregate
from algorithm.autoencoder import AutoEncoder
from algorithm.evt import EVT
from algorithm.ksigma import KSigma
from algorithm.lstm import LSTM
from algorithm.matrix_profile import MatrixProfile
from algorithm.mlp import MLP
from algorithm.random_detector import Random
from algorithm.random_trigger import RandomTrigger
from algorithm.sr import SR
from config import ANOMALY_SCORE_COLUMN, THRESHOLD_COLUMN, TRAIN_TIME, TEST_TIME, PULL_TIME, LABEL_COLUMN, \
    TIME_END_COLUMN, NO_NEED_TO_RENAME_COLUMNS
from data_prepare.dataset import Dataset
from data_prepare.raw_time_series import RawTimeSeries
from data_prepare.result_time_series import ResultTimeSeries
from detector.detector import Detector, MultivariateDetector
from detector.fit import FitMode, SupervisedFit, UnsupervisedFit
from detector.predict import PredictMode, OfflinePredict, StreamingPredict, TriggerPredict
from evaluate.eval import eval
from threshold.threshold import Threshold
from transform.transform import Transform
from utils.preprocess import sliding
from utils.timer import timer, time_on, get_time


class TaskExecutor:
    @staticmethod
    def exec(data: Union[RawTimeSeries, Dataset, str], detector: Union[Detector, list[Detector]], detector_name: str,
             transform: Transform = None, aggregate: Aggregate = MaxAggregate(), threshold: Threshold = None,
             streaming_batch_size=1, window_size=20, anomaly_ratio=0.10):
        @timer(TRAIN_TIME)
        def supervised_fit(raw_time_series: RawTimeSeries, supervised_fitter: SupervisedFit):
            train_data, label = raw_time_series.get_train_data()
            return supervised_fitter.fit(train_data, label)

        @timer(TRAIN_TIME)
        def unsupervised_fit(raw_time_series: RawTimeSeries, unsupervised_fitter: UnsupervisedFit):
            train_data, _ = raw_time_series.get_train_data()
            return unsupervised_fitter.fit(train_data)

        @timer(TRAIN_TIME)
        def unwanted_fit(raw_time_series: RawTimeSeries, unwanted_fitter: Detector):
            return None

        @timer(TEST_TIME)
        def offline_predict(raw_time_series: RawTimeSeries, offline_predictor: OfflinePredict):
            test_data, _ = raw_time_series.get_test_data()
            return offline_predictor.predict(test_data)

        @timer(TEST_TIME)
        def stream_predict(raw_time_series: RawTimeSeries, stream_predictor: StreamingPredict):
            train_data, _ = raw_time_series.get_train_data()
            stream_predictor.init(train_data)
            test_data, _ = raw_time_series.get_test_data()
            return [stream_predictor.predict(x) for x in sliding(test_data, streaming_batch_size)]

        @timer(TEST_TIME)
        def trigger_predict(raw_time_series: RawTimeSeries, trigger_predictor: TriggerPredict):
            # 直接在这一步将结果转成dataframe
            detect_ranges = raw_time_series.select_detect_range(window_size=window_size, anomaly_ratio=anomaly_ratio,
                                                                is_disjoint=True)
            train_data, _ = raw_time_series.get_train_data()
            test_data, _ = raw_time_series.get_test_data()
            history_data = np.r_[train_data, test_data]
            detect_task = {i[0]: i for i in detect_ranges}
            test_timestamp = raw_time_series.get_test_timestamp()
            result = []
            for i in range(len(test_data)):
                if i not in detect_task:
                    result.append({})
                else:
                    interval = detect_task[i]

                    @timer(PULL_TIME)
                    def pull_data(points: int) -> np.ndarray:
                        # TODO 改成真正从数据库拉数据
                        return history_data[
                               raw_time_series.train_data_len + interval[1] - points: raw_time_series.train_data_len +
                                                                                      interval[1] + 1, :]

                    rs = trigger_predictor.predict(pull_data)
                    if not isinstance(rs, dict):
                        rs = {ANOMALY_SCORE_COLUMN: rs}
                    # rs['timeBegin'] = test_timestamp[interval[0]]
                    rs[TIME_END_COLUMN] = test_timestamp[interval[1]] // (10 ** 6)
                    rs[LABEL_COLUMN] = interval[2]
                    result.append(rs)
            return pd.DataFrame.from_records(result)

        fit_mode2executor = {
            FitMode.Supervised: supervised_fit,
            FitMode.Unsupervised: unsupervised_fit,
            FitMode.Unwanted: unwanted_fit
        }

        predict_mode2executor = {
            PredictMode.Offline: offline_predict,
            PredictMode.Stream: stream_predict,
            PredictMode.Trigger: trigger_predict
        }

        def run(ts: RawTimeSeries):
            fit_method = fit_mode2executor[detector.fit_mode]
            predict_method = predict_mode2executor[detector.predict_mode]
            time_on()

            def parse(result, is_test):
                if result is None:
                    dataframe = pd.DataFrame()
                else:
                    if isinstance(result, pd.DataFrame):
                        dataframe = result
                    elif isinstance(result, list):
                        if isinstance(result[0], float):
                            dataframe = pd.DataFrame(data={ANOMALY_SCORE_COLUMN: result})
                        elif isinstance(result[0], dict):
                            if is_test and ANOMALY_SCORE_COLUMN not in result[0]:
                                raise ValueError('must have a key named score')
                            dataframe = pd.DataFrame.from_records(result)
                        else:
                            raise ValueError(f'element of list must be float or dict! get {type(result[0])}')
                    else:
                        raise ValueError(f'result must be dataframe or list! get {type(result)}')
                return dataframe

            def gen_result_df(real_ts: RawTimeSeries, dt: Detector, columns_prefix: str) -> (
                    pd.DataFrame, list[float], list[float]):
                logging.info(f'running {dt.__class__.__name__} for {real_ts.ts_name} of {real_ts.ds_name}...')
                df = parse(fit_method(real_ts, dt), is_test=False)
                df.index = ts.data.index[:len(df)]
                train_score_list = [] if ANOMALY_SCORE_COLUMN not in df.columns else df[ANOMALY_SCORE_COLUMN].tolist()
                df.rename(
                    columns={col: columns_prefix + col for col in df.columns if col not in NO_NEED_TO_RENAME_COLUMNS},
                    inplace=True)
                tf = parse(predict_method(real_ts, dt), is_test=True)
                tf.index = ts.data.index[-len(tf):]
                test_score_list = tf[ANOMALY_SCORE_COLUMN].tolist()
                tf.rename(
                    columns={col: columns_prefix + col for col in tf.columns if col not in NO_NEED_TO_RENAME_COLUMNS},
                    inplace=True)
                return pd.concat([df, tf]), train_score_list, test_score_list

            if not isinstance(detector, list):
                detector.save(detector_name)
            if isinstance(detector, Detector) and (isinstance(detector, MultivariateDetector) or ts.dim_num == 1):
                result_df, train_score, test_score = gen_result_df(ts, detector, '')
            else:
                result_df = pd.DataFrame()
                train_score = []
                test_score = []

                def append(temp_result_df):
                    temp, train, test = temp_result_df
                    if transform is not None:
                        train, test = transform.transform(train, test)
                    nonlocal train_score, test_score, result_df
                    if len(train_score) == 0:
                        train_score = [[] for _ in range(len(train))]
                    for i, x in enumerate(train):
                        train_score[i].append(x)
                    if len(test_score) == 0:
                        test_score = [[] for _ in range(len(test))]
                    for i, x in enumerate(test):
                        test_score[i].append(x)
                    result_df = pd.concat([result_df, temp], axis=1)

                if isinstance(detector, list):
                    for d in detector:
                        append(gen_result_df(real_ts=ts, dt=d, columns_prefix=d.name + '_'))
                else:
                    for col in tqdm(ts.get_columns(), desc=f'\t Running every dim for {ts.ts_name}'):
                        append(gen_result_df(real_ts=ts.get_column_data(column_name=col), dt=detector,
                                             columns_prefix=col + '_'))
                if aggregate is None:
                    raise ValueError('ensemble method / univariate to multivariate should have a aggregate method')
                train_score, test_score = aggregate.aggregate(train=train_score, test=test_score)
                result_df[ANOMALY_SCORE_COLUMN] = train_score + test_score
                result_df = result_df.loc[:, ~result_df.columns.duplicated()]
            if threshold is not None:
                th = threshold.threshold(train_score, test_score)
                result_df[THRESHOLD_COLUMN] = th
            elif THRESHOLD_COLUMN in result_df.columns:
                th = result_df[THRESHOLD_COLUMN].tolist()
            else:
                th = None
            if isinstance(detector, TriggerPredict):
                for column in result_df.columns:
                    if column.endswith(LABEL_COLUMN):
                        eval_result = eval(test_score, result_df[column].tolist(), th)
                        break
                else:
                    logging.info(f"Can not find label for {detector_name} of {ts.ts_name}")
                    eval_result = {}
            else:
                eval_result = eval(test_score, ts.get_test_data()[1].tolist(), th)
            ResultTimeSeries(data=result_df, ds_name=ts.ds_name, ts_name=ts.ts_name,
                             detector_name=detector_name, eval_result=eval_result, cost_time=get_time()).save()

        if isinstance(data, RawTimeSeries):
            run(data)
        else:
            if isinstance(data, str):
                dataset = Dataset.load(data)
            else:
                dataset = data
            for time_series in tqdm(dataset.ts, desc=f'Running time series of {dataset.name}'):
                run(time_series)


def run_univariate_algorithm(algorithms, univariate_datasets=['Yahoo', 'KPI', 'Industry'],
                             multivariate_datasets=['SMD', 'JumpStarter', 'SKAB']):
    for detector in algorithms:
        for dataset in univariate_datasets:
            try:
                TaskExecutor.exec(data=dataset, detector=detector, detector_name=detector.__class__.__name__)
            except Exception as e:
                print(e, detector.__class__.__name__, dataset)
        for dataset in multivariate_datasets:
            try:
                TaskExecutor.exec(data=dataset, detector=detector, detector_name=f'Max{detector.__class__.__name__}',
                                  aggregate=MaxAggregate())
            except Exception as e:
                print(e, detector.__class__.__name__, dataset)


def test_mp():
    for batch_size in [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200]:
        mp_detector = MatrixProfile(batch_size)
        raw_time_series = RawTimeSeries.load("Yahoo@synthetic_1")
        TaskExecutor.exec(raw_time_series, detector=mp_detector, detector_name=f"mp_{batch_size}")


def test_sr():
    sr_detector = SR()
    raw_time_series = RawTimeSeries.load("Yahoo@synthetic_1")
    TaskExecutor.exec(raw_time_series, detector=sr_detector, detector_name=f"test_sr")


def test_metric():
    TaskExecutor.exec(RawTimeSeries.load("Yahoo@synthetic_1"), detector=MLP(window_size=30),
                      detector_name='test_mlp_once')

    TaskExecutor.exec("Yahoo", detector=MLP(window_size=30), detector_name='test_mlp')


def test_ksigma():
    random_trigger = KSigma()
    raw_time_series = RawTimeSeries.load("Yahoo@synthetic_1")
    TaskExecutor.exec(raw_time_series, detector=random_trigger, detector_name=f"test_ksgima")


def test_trigger():
    random_trigger = RandomTrigger()
    raw_time_series = RawTimeSeries.load("Yahoo@synthetic_1")
    TaskExecutor.exec(raw_time_series, detector=random_trigger, detector_name=f"test_random_trigger")
    raw_time_series = RawTimeSeries.load("SMD@machine-1-1")
    TaskExecutor.exec(raw_time_series, detector=random_trigger, detector_name=f"max_test_random_trigger")


if __name__ == '__main__':
    # run_univariate_algorithm(algorithms=[
    #     Random(),
    #     EVT(),
    #     MatrixProfile(),
    #     SR(),
    #     MLP(window_size=30),
    #     AutoEncoder(window_size=30, z_dim=10),
    #     LSTM(window_size=30, batch_size=16, hidden_size=10)
    # ],
    #     univariate_datasets=['Yahoo', 'KPI'],
    #     multivariate_datasets=['SMD', 'JumpStarter', 'SKAB'])
    # test_metric()
    # test_mp()
    # test_sr()
    # test_trigger()
    # test_ksigma()
    run_univariate_algorithm([KSigma()])
    # run_univariate_algorithm(algorithms=[MatrixProfile(20),
    #                                      SR()],
    #                          univariate_datasets=[],  # ['Yahoo', 'Industry'],
    #                          multivariate_datasets=['SMD', 'JumpStarter', 'SKAB'])
