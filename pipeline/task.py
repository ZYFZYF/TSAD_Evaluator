# @Time    : 2022/3/3 21:51
# @Author  : ZYF
from typing import Union

from dataset.dataset import Dataset
from dataset.raw_time_series import RawTimeSeries
from detector.detector import Detector
from detector.fit import FitMode
from detector.predict import PredictMode


def supervised_fit(time_series: RawTimeSeries, detector: Detector):
    (data, label) = time_series.get_train_data()
    detector.fit(data, label)


def unsupervised_fit(time_series: RawTimeSeries, detector: Detector):
    (data, _) = time_series.get_train_data()
    detector.fit(data)


def unwanted_fit(time_series: RawTimeSeries, detector: Detector):
    ...


def offline_predict(time_series: RawTimeSeries, detector: Detector):
    (data, label) = time_series.get_train_data()
    predict = detector.predict(data)
    # TODO 评估以及存储序列化的东西


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
            fit_method(time_series=ts, detector=detector)
            predict_method(time_series=time_series, detector=detector)

        if isinstance(data, RawTimeSeries):
            run(data)
        else:
            if isinstance(data, str):
                dataset = Dataset.load(data)
            else:
                dataset = data
            for time_series in dataset.ts:
                run(time_series)
