# @Time    : 2022/3/6 10:25
# @Author  : ZYF
from typing import Optional

import numpy as np

from evaluate.metric_with_threshold import metric_with_threshold_list
from evaluate.metric_without_threshold import metric_without_threshold_list


def eval(pred: list[float], label: list[float], threshold: Optional[list[float]]):
    print(type(pred), type(label), len(pred), len(label))
    if len(pred) < len(label):
        pred = np.array([0.0] * (len(label) - len(pred)) + pred)
    if threshold is not None:
        predict = [1 if s > th else 0 for (s, th) in zip(pred, threshold)]
        return {metric.__name__: metric.score(predict, label) for metric in metric_with_threshold_list}
    else:
        return {metric.__name__: metric.score(pred, label) for metric in metric_without_threshold_list}
