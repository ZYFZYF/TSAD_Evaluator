# @Time    : 2022/3/6 10:25
# @Author  : ZYF
import numpy as np

from evaluate.metric_with_threshold import metric_with_threshold_list
from evaluate.metric_without_threshold import metric_without_threshold_list


def evaluate(pred: np.ndarray, label: np.ndarray, threshold):
    if threshold:
        predict = [1 if s > threshold else 0 for s in pred]
        return {metric.__name__: metric.score(predict, label) for metric in metric_with_threshold_list}
    else:
        return {metric.__name__: metric.score(pred, label) for metric in metric_without_threshold_list}
