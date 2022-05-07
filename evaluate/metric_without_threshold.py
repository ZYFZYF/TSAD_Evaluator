# @Time    : 2022/3/10 23:38
# @Author  : ZYF
import abc
import random
from typing import Generator

from evaluate.utils import safeFloatDivide, getPRF, extractEvent
from evaluate.metric_with_threshold import Fpa1, Fc1, F1, Fd1

metric_without_threshold_list = []


class MetricWithoutThreshold(abc.ABCMeta):

    def __init_subclass__(mcs, **kwargs):
        metric_without_threshold_list.append(mcs)

    @classmethod
    @abc.abstractmethod
    def score(mcs, predict: list[float], label: list[float]) -> float:
        ...


class ScoreSearcher(abc.ABCMeta):
    @classmethod
    @abc.abstractmethod
    def getScore(mcs, predict: list[float], label: list[float]) -> Generator[tuple, int, None]:
        ...

    # 把每个点从高到低进行召回
    @classmethod
    def searchAllThreshold(mcs, predict: list[float], label: list[float]) -> list[tuple]:
        iterator = mcs.getScore(predict, label)
        iterator.send(None)
        n = len(label)
        kLargestIndex = [i for i in range(n)]
        kLargestIndex = sorted(kLargestIndex, key=lambda x: predict[x], reverse=True)
        return [iterator.send(kLargestIndex[i]) for i in range(n)]


class SearchF1(ScoreSearcher):
    @classmethod
    def getScore(mcs, predict: list[float], label: list[float]) -> Generator[tuple, int, None]:
        TP = 0
        FN = sum(label)
        FP = 0
        while True:
            leftMaxIndex = yield getPRF(TP, FP, FN)
            if label[leftMaxIndex] == 0:
                FP += 1
            else:
                TP += 1
                FN -= 1


def areaUnder(score: list[tuple[float, float]]) -> float:
    area = 0
    lastX = 0
    for xVal, yVal in score:
        area += (xVal - lastX) * yVal
        lastX = xVal
    return area


class SearchFpa1(ScoreSearcher):
    @classmethod
    def getScore(mcs, predict: list[float], label: list[float]) -> Generator[tuple, int, None]:
        head, tail, _ = extractEvent(label)
        TP = 0
        FN = sum(label)
        FP = 0
        labeled_event = set()
        while True:
            leftMaxIndex = yield getPRF(TP, FP, FN)
            if label[leftMaxIndex] == 0:
                FP += 1
            else:
                event = (head[leftMaxIndex], tail[leftMaxIndex])
                if event not in labeled_event:
                    labeled_event.add(event)
                    TP += event[1] - event[0] + 1
                    FN -= event[1] - event[0] + 1


class SearchFc1(ScoreSearcher):
    @classmethod
    def getScore(mcs, predict: list[float], label: list[float]) -> Generator[tuple, int, None]:
        head, tail, event_count = extractEvent(label)
        TP_t = 0
        FP_t = 0
        TP_e = 0
        FN_e = event_count
        labeled_event = set()
        while True:
            recall = safeFloatDivide(TP_e, TP_e + FN_e)
            precision = safeFloatDivide(TP_t, TP_t + FP_t)
            f1_score = safeFloatDivide(recall * precision * 2, recall + precision)
            leftMaxIndex = yield precision, recall, f1_score
            if label[leftMaxIndex] == 0:
                FP_t += 1
            else:
                event = (head[leftMaxIndex], tail[leftMaxIndex])
                if event not in labeled_event:
                    labeled_event.add(event)
                    TP_e += 1
                    FN_e -= 1
                TP_t += 1


class SearchFd1(ScoreSearcher):
    @classmethod
    def getScore(mcs, predict: list[float], label: list[float]) -> Generator[tuple, int, None]:
        head, tail, event_count = extractEvent(label)
        TP = 0
        FP = 0
        delay = dict()
        sumDelay = event_count * 1.0
        for i in head:
            if i != -1:
                delay[i] = 1
        while True:
            recall = safeFloatDivide(event_count - sumDelay, event_count)
            precision = safeFloatDivide(TP, TP + FP)
            f1_score = safeFloatDivide(recall * precision * 2, recall + precision)
            leftMaxIndex = yield precision, recall, f1_score
            if label[leftMaxIndex] == 0:
                FP += 1
            else:
                TP += 1
                d = (leftMaxIndex - head[leftMaxIndex]) / (tail[leftMaxIndex] - head[leftMaxIndex] + 1)
                if d < delay[head[leftMaxIndex]]:
                    sumDelay += d - delay[head[leftMaxIndex]]
                    delay[head[leftMaxIndex]] = d


class BestF1(MetricWithoutThreshold):

    @classmethod
    def score(mcs, predict: list[float], label: list[float]) -> float:
        return max([score[2] for score in SearchF1.searchAllThreshold(predict, label)])


class BestFpa1(MetricWithoutThreshold):

    @classmethod
    def score(mcs, predict: list[float], label: list[float]) -> float:
        return max([score[2] for score in SearchFpa1.searchAllThreshold(predict, label)])


class BestFc1(MetricWithoutThreshold):
    @classmethod
    def score(mcs, predict: list[float], label: list[float]) -> float:
        return max([score[2] for score in SearchFc1.searchAllThreshold(predict, label)])


class BestFd1(MetricWithoutThreshold):
    @classmethod
    def score(mcs, predict: list[float], label: list[float]) -> float:
        return max([score[2] for score in SearchFd1.searchAllThreshold(predict, label)])


class AUPRC(MetricWithoutThreshold):
    @classmethod
    def score(mcs, predict: list[float], label: list[float]) -> float:
        return areaUnder([(r, p) for p, r, _ in SearchF1.searchAllThreshold(predict, label)])


class AUPRCpa(MetricWithoutThreshold):
    @classmethod
    def score(mcs, predict: list[float], label: list[float]) -> float:
        return areaUnder([(r, p) for p, r, _ in SearchFpa1.searchAllThreshold(predict, label)])


class AUPRCc(MetricWithoutThreshold):
    @classmethod
    def score(mcs, predict: list[float], label: list[float]) -> float:
        return areaUnder([(r, p) for p, r, _ in SearchFc1.searchAllThreshold(predict, label)])


class AUPRCd(MetricWithoutThreshold):
    @classmethod
    def score(mcs, predict: list[float], label: list[float]) -> float:
        return areaUnder([(r, p) for p, r, _ in SearchFd1.searchAllThreshold(predict, label)])


class TrivialBestF1:
    @classmethod
    def score(cls, predict: list[float], label: list[float]) -> float:
        return max([F1.score([0 if pred < th else 1 for pred in predict], label) for th in predict])


class TrivialBestFpa1:
    @classmethod
    def score(cls, predict: list[float], label: list[float]) -> float:
        return max([Fpa1.score([0 if pred < th else 1 for pred in predict], label) for th in predict])


class TrivialBestFc1:
    @classmethod
    def score(cls, predict: list[float], label: list[float]) -> float:
        return max([Fc1.score([0 if pred < th else 1 for pred in predict], label) for th in predict])


class TrivialBestFd1:
    @classmethod
    def score(cls, predict: list[float], label: list[float]) -> float:
        return max([Fd1.score([0 if pred < th else 1 for pred in predict], label) for th in predict])


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

        for metric in metric_without_threshold_list:
            print(metric.__name__, metric.score(x, y))


        def make_sure_equal(raw_result, fast_result):
            print(raw_result, fast_result)
            assert abs(raw_result - fast_result) < 1e-5


        make_sure_equal(TrivialBestF1.score(x, y), BestF1.score(x, y))
        make_sure_equal(TrivialBestFpa1.score(x, y), BestFpa1.score(x, y))
        make_sure_equal(TrivialBestFc1.score(x, y), BestFc1.score(x, y))
        make_sure_equal(TrivialBestFd1.score(x, y), BestFd1.score(x, y))
