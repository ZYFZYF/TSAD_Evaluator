# @Time    : 2022/3/10 23:38
# @Author  : ZYF
import abc

metric_without_threshold_list = []


class MetricWithoutThreshold(abc.ABCMeta):

    def __init_subclass__(mcs, **kwargs):
        metric_without_threshold_list.append(mcs)

    @classmethod
    @abc.abstractmethod
    def score(mcs, predict, label):
        ...


from evaluate.metric_with_threshold import F1


class BestF1(MetricWithoutThreshold):
    @classmethod
    def score(mcs, predict, label):
        return max([F1.score([1 if s > th else 0 for s in predict], label) for th in predict])


if __name__ == '__main__':
    x = [0, 0.2, 0.3, 0.5, 0.6, 0.1]
    y = [0, 1, 1, 1, 0, 1]
    print(metric_without_threshold_list)
    for metric in metric_without_threshold_list:
        print(metric.__name__, metric.score(x, y))
