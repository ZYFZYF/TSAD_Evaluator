# @Time    : 2022/3/6 10:25
# @Author  : ZYF
import numpy as np

from evaluate.metric_with_threshold import metric_with_threshold_list
from evaluate.metric_without_threshold import metric_without_threshold_list


def eval(pred: np.ndarray, label: np.ndarray, threshold):
    print(type(pred), type(label))
    if pred.shape[0] < label.shape[0]:
        pred = np.array([0.0] * (label.shape[0] - pred.shape[0]) + pred.tolist())
    if threshold is not None:
        predict = [1 if s > th else 0 for (s, th) in zip(pred, threshold)]
        return {metric.__name__: metric.score(predict, label) for metric in metric_with_threshold_list}
    else:
        return {metric.__name__: metric.score(pred, label) for metric in metric_without_threshold_list}
