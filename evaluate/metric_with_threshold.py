# @Time    : 2022/3/10 23:14
# @Author  : ZYF
import abc

from sklearn.metrics import precision_score, recall_score, f1_score

metric_with_threshold_list = []


class MetricWithThreshold(abc.ABCMeta):

    def __init_subclass__(mcs, **kwargs):
        metric_with_threshold_list.append(mcs)

    @classmethod
    @abc.abstractmethod
    def score(mcs, predict, label):
        ...


class Precision(MetricWithThreshold):
    @classmethod
    def score(mcs, predict, label):
        return precision_score(label, predict)


class Recall(MetricWithThreshold):
    @classmethod
    def score(mcs, predict, label):
        return recall_score(label, predict)


class F1(MetricWithThreshold):

    @classmethod
    def score(mcs, predict, label):
        return f1_score(label, predict)


if __name__ == '__main__':
    x = [0, 1, 0, 1, 1, 0]
    y = [0, 1, 1, 1, 0, 1]
    print(Precision.score(x, y))
    print(metric_with_threshold_list)
    for metric in metric_with_threshold_list:
        print(metric.__name__, metric.score(x, y))
