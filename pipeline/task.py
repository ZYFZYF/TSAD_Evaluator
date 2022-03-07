# @Time    : 2022/3/3 21:51
# @Author  : ZYF
from typing import Union

import pandas as pd

from dataset.dataset import Dataset
from dataset.raw_time_series import RawTimeSeries
from dataset.result_time_series import ResultTimeSeries
from detector.detector import Detector
from detector.fit import FitMode
from detector.predict import PredictMode
from evaluate.evaluate import evaluate


def supervised_fit(time_series: RawTimeSeries, detector: Detector):
    (data, label) = time_series.get_train_data()
    return detector.fit(data, label)


def unsupervised_fit(time_series: RawTimeSeries, detector: Detector):
    (data, _) = time_series.get_train_data()
    return detector.fit(data)


def unwanted_fit(time_series: RawTimeSeries, detector: Detector):
    ...


def offline_predict(time_series: RawTimeSeries, detector: Detector):
    (data, label) = time_series.get_train_data()
    return detector.predict(data)


def stream_predict(time_series: RawTimeSeries, detector: Detector):
    # TODO 流式评估
    ...


def trigger_predict(time_series: RawTimeSeries, detector: Detector):
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


class TaskExecutor:
    @staticmethod
    def exec(data: Union[RawTimeSeries, Dataset, str], detector: Detector):
        def run(ts: RawTimeSeries):
            fit_method = fit_mode2executor[detector.fit_mode]
            predict_method = predict_mode2executor[detector.predict_mode]

            def parse(result):
                if isinstance(result, pd.Dataframe):
                    dataframe = result
                elif isinstance(result, list):
                    if isinstance(result[0], float):
                        dataframe = pd.DataFrame(data={'score': result})
                    elif isinstance(result[0], dict):
                        if 'score' not in result[0]:
                            raise ValueError('must have a key named score')
                        dataframe = pd.DataFrame.from_records(result)
                    else:
                        raise ValueError('element of list must be float or dict')
                else:
                    raise ValueError('result must be dataframe or list!')
                return dataframe

            df = parse(fit_method(time_series=ts, detector=detector))
            df.index = ts.data.index[:len(df)]
            tf = parse(predict_method(time_series=time_series, detector=detector))
            tf.index = ts.data.index[-len(tf):]
            eval_result = evaluate(tf['score'].tolist(), ts.get_test_data()[1])
            ResultTimeSeries(data=pd.concat([df, tf]), ds_name=ts.ds_name, ts_name=ts.ts_name,
                             detector_name=detector.name, eval_result=eval_result).save()

        if isinstance(data, RawTimeSeries):
            run(data)
        else:
            if isinstance(data, str):
                dataset = Dataset.load(data)
            else:
                dataset = data
            for time_series in dataset.ts:
                run(time_series)
