# @Time    : 2022/3/3 21:51
# @Author  : ZYF
import logging
from typing import Union

import pandas as pd

from aggregate.aggregate import Aggregate
from config import ANOMALY_SCORE_COLUMN, THRESHOLD_COLUMN
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


class TaskExecutor:
    @staticmethod
    def exec(data: Union[RawTimeSeries, Dataset, str], detector: Union[Detector, list[Detector]], detector_name: str,
             transform: Transform = None, aggregate: Aggregate = None, threshold: Threshold = None,
             streaming_batch_size=1):

        def supervised_fit(raw_time_series: RawTimeSeries, supervised_fitter: SupervisedFit):
            train_data, label = raw_time_series.get_train_data()
            return supervised_fitter.fit(train_data, label)

        def unsupervised_fit(raw_time_series: RawTimeSeries, unsupervised_fitter: UnsupervisedFit):
            train_data, _ = raw_time_series.get_train_data()
            return unsupervised_fitter.fit(train_data)

        def unwanted_fit(raw_time_series: RawTimeSeries, unwanted_fitter: Detector):
            return None

        def offline_predict(raw_time_series: RawTimeSeries, offline_predictor: OfflinePredict):
            test_data, _ = raw_time_series.get_test_data()
            return offline_predictor.predict(test_data)

        def stream_predict(raw_time_series: RawTimeSeries, stream_predictor: StreamingPredict):
            train_data, _ = raw_time_series.get_train_data()
            stream_predictor.init(train_data)
            test_data, _ = raw_time_series.get_test_data()
            return [stream_predictor.predict(x) for x in sliding(test_data, streaming_batch_size)]

        def trigger_predict(raw_time_series: RawTimeSeries, trigger_predictor: TriggerPredict):
            # TODO 触发式评估
            ...

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
                            raise ValueError('element of list must be float or dict')
                    else:
                        raise ValueError('result must be dataframe or list!')
                return dataframe

            def gen_result_df(real_ts: RawTimeSeries, dt: Detector, columns_prefix: str) -> (
                    pd.DataFrame, list[float], list[float]):
                logging.info(f'running {dt.__class__.__name__} for {real_ts.ts_name} of {real_ts.ds_name}...')
                df = parse(fit_method(time_series=real_ts, detector=dt), is_test=False)
                df.index = ts.data.index[:len(df)]
                df.rename(columns={col: columns_prefix + col for col in df.columns}, inplace=True)
                tf = parse(predict_method(time_series=real_ts, detector=dt), is_test=True)
                tf.index = ts.data.index[-len(tf):]
                tf.rename(columns={col: columns_prefix + col for col in tf.columns}, inplace=True)
                return pd.concat([df, tf]), (
                    [] if ANOMALY_SCORE_COLUMN not in df.columns else df[ANOMALY_SCORE_COLUMN].tolist()), tf[
                           ANOMALY_SCORE_COLUMN].tolist()

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
                    for col in ts.get_columns():
                        append(gen_result_df(real_ts=ts.get_column_data(column_name=col), dt=detector,
                                             columns_prefix=col + '_'))
                if aggregate is None:
                    raise ValueError('ensemble method should have a aggregate method')
                train_score, test_score = aggregate.aggregate(train=train_score, test=test_score)
                result_df[ANOMALY_SCORE_COLUMN] = train_score + test_score
            if threshold is not None:
                th = threshold.threshold(train_score, test_score)
                result_df[THRESHOLD_COLUMN] = th
            else:
                th = None
            eval_result = eval(test_score, ts.get_test_data()[1], th)
            ResultTimeSeries(data=result_df, ds_name=ts.ds_name, ts_name=ts.ts_name,
                             detector_name=detector_name, eval_result=eval_result).save()

        if isinstance(data, RawTimeSeries):
            run(data)
        else:
            if isinstance(data, str):
                dataset = Dataset.load(data)
            else:
                dataset = data
            for time_series in dataset.ts:
                run(time_series)


if __name__ == '__main__':
    # test_detector = RandomDetector()
    # print(dir(test_detector))
    # test_ts = RawTimeSeries.load('Yahoo@synthetic_1')
    # TaskExecutor.exec(data=test_ts, detector=test_detector, detector_name='test_random')
    # from detector.autoencoder import AutoEncoder
    #
    # ae_detector = AutoEncoder(window_size=60, z_dim=10)
    # test_ts = RawTimeSeries.load('Yahoo@synthetic_1')
    # TaskExecutor.exec(data=test_ts, detector=ae_detector, detector_name='test_autoencoder')

    # from detector.lstm import LSTM
    #
    # lstm_detector = LSTM(window_size=50, batch_size=16, hidden_size=10)
    # test_ts = RawTimeSeries.load('Yahoo@synthetic_1')
    # TaskExecutor.exec(data=test_ts, detector=lstm_detector, detector_name='test_lstm')

    from algorithm.mlp import MLP

    lstm_detector = MLP(window_size=20)
    test_ts = RawTimeSeries.load('Yahoo@synthetic_11')
    TaskExecutor.exec(data=test_ts, detector=lstm_detector, detector_name='test_mlp')
