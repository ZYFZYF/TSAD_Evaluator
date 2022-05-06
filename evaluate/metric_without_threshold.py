# @Time    : 2022/3/10 23:38
# @Author  : ZYF
import abc
import random

from config import EPS
from evaluate.metric_with_threshold import Fpa, Fc1, F1

metric_without_threshold_list = []


class MetricWithoutThreshold(abc.ABCMeta):

    def __init_subclass__(mcs, **kwargs):
        metric_without_threshold_list.append(mcs)

    @classmethod
    @abc.abstractmethod
    def score(mcs, predict: list[float], label: list[float]) -> float:
        ...


class BestF1(MetricWithoutThreshold):
    @classmethod
    def score(mcs, predict: list[float], label: list[float]) -> float:
        best_f1_score = 0.0
        n = len(label)
        order = [i for i in range(n)]
        order = sorted(order, key=lambda x: predict[x], reverse=True)
        TP = 0
        TN = n - sum(label)
        FN = sum(label)
        FP = 0
        for i in range(n):
            post = order[i]
            if label[post] == 0:
                TN -= 1
                FP += 1
            else:
                TP += 1
                FN -= 1
            recall = 1.0 * TP / (TP + FN + EPS)
            precision = 1.0 * TP / (TP + FP + EPS)
            f1_score = 2.0 * recall * precision / (recall + precision + EPS)
            if f1_score > best_f1_score:
                best_f1_score = f1_score
        return best_f1_score


class BestFpa(MetricWithoutThreshold):

    @classmethod
    def score(mcs, predict: list[float], label: list[float]) -> float:
        best_fpa_score = 0.0
        n = len(label)
        head = [0] * n
        tail = [0] * n
        le = 0
        while le < n:
            while le < n and label[le] != 1:
                le += 1
            if le == n:
                break
            ri = le
            while ri + 1 < n and label[ri + 1] == 1:
                ri += 1
            for i in range(le, ri + 1):
                head[i] = le
                tail[i] = ri
            le = ri + 1
        order = [i for i in range(n)]
        order = sorted(order, key=lambda x: predict[x], reverse=True)
        TP = 0
        TN = n - sum(label)
        FN = sum(label)
        FP = 0
        labeled_event = set()
        for i in range(n):
            post = order[i]
            if label[post] == 0:
                TN -= 1
                FP += 1
            else:
                if (event := (head[post], tail[post])) not in labeled_event:
                    labeled_event.add(event)
                    TP += tail[post] - head[post] + 1
                    FN -= tail[post] - head[post] + 1
            recall = 1.0 * TP / (TP + FN + EPS)
            precision = 1.0 * TP / (TP + FP + EPS)
            f1_score = 2.0 * recall * precision / (recall + precision + EPS)
            if f1_score > best_fpa_score:
                best_fpa_score = f1_score
        return best_fpa_score


class BestFc1(MetricWithoutThreshold):

    @classmethod
    def score(mcs, predict: list[float], label: list[float]) -> float:
        best_fc1_score = 0.0
        n = len(label)
        head = [0] * n
        tail = [0] * n
        event_count = 0
        le = 0
        while le < n:
            while le < n and label[le] != 1:
                le += 1
            if le == n:
                break
            event_count += 1
            ri = le
            while ri + 1 < n and label[ri + 1] == 1:
                ri += 1
            for i in range(le, ri + 1):
                head[i] = le
                tail[i] = ri
            le = ri + 1
        order = [i for i in range(n)]
        order = sorted(order, key=lambda x: predict[x], reverse=True)
        TP_t = 0
        FP_t = 0
        TP_e = 0
        FN_e = event_count
        labeled_event = set()
        for i in range(n):
            post = order[i]
            if label[post] == 0:
                FP_t += 1
            else:
                if (event := (head[post], tail[post])) not in labeled_event:
                    labeled_event.add(event)
                    TP_e += 1
                    FN_e -= 1
                TP_t += 1
            recall = 1.0 * TP_e / (TP_e + FN_e + EPS)
            precision = 1.0 * TP_t / (TP_t + FP_t + EPS)
            f1_score = 2.0 * recall * precision / (recall + precision + EPS)
            if f1_score > best_fc1_score:
                best_fc1_score = f1_score
        return best_fc1_score


class TrivialBestF1:
    @classmethod
    def score(cls, predict: list[float], label: list[float]) -> float:
        return max([F1.score([0 if pred < th else 1 for pred in predict], label) for th in predict])


class TrivialBestFpa:
    @classmethod
    def score(cls, predict: list[float], label: list[float]) -> float:
        return max([Fpa.score([0 if pred < th else 1 for pred in predict], label) for th in predict])


class TrivialBestFc1:
    @classmethod
    def score(cls, predict: list[float], label: list[float]) -> float:
        return max([Fc1.score([0 if pred < th else 1 for pred in predict], label) for th in predict])


if __name__ == '__main__':
    x = [0, 0.2, 0.3, 0.5, 0.6, 0.1]
    y = [0, 1, 1, 1, 0, 1]
    print(metric_without_threshold_list)
    for metric in metric_without_threshold_list:
        print(metric.__name__, metric.score(x, y))
    for j in range(20):
        N = 1000
        R = 0.1
        x = [random.random() for i in range(N)]
        if j < 10:
            y = [0 if random.random() > R else 1 for i in range(N)]
            print(f"随机分布的异常，异常点数{sum(y)}/{N}")
        else:
            num = random.randint(1, 6)
            M = int(N * R / num)
            y = [0 for i in range(N)]
            for i in range(num):
                while True:
                    j = random.randint(0, N)
                    if j > 0 and j + M < N and y[j - 1] == 0 and y[j + M] == 0:
                        for k in range(j, j + M):
                            y[k] = 1
                        break
            print(f"尽可能长的异常，异常点数{sum(y)}/{N}，分成{num}段，每段{M}个点")


        def make_sure_equal(raw_result, fast_result):
            print(raw_result, fast_result)
            assert abs(raw_result - fast_result) < 1e-5


        make_sure_equal(TrivialBestF1.score(x, y), BestF1.score(x, y))
        make_sure_equal(TrivialBestFpa.score(x, y), BestFpa.score(x, y))
        make_sure_equal(TrivialBestFc1.score(x, y), BestFc1.score(x, y))
